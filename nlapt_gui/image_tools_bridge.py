"""Async bridge for alpha-flatten, duplicate scan, quarantine, and restore."""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import QObject, QThreadPool, Signal

from nlapt.core.errors import NLaptError
from nlapt.diagnostics import get_logger
from nlapt.images.alpha import FlattenReport, FlattenSpec, flatten_alpha, pick_color
from nlapt.images.backup import ImageBackupManager
from nlapt.images.hashing import DuplicateGroup, fingerprint, group_similar

from nlapt_gui.controller import TOAST_ERR, TOAST_INFO, TOAST_NO_SELECTION, TOAST_OK, TOAST_WARN, AppController
from nlapt_gui.workers import run_async

_LOGGER = get_logger(__name__)

TOAST_NO_DATASET = "请先打开数据集"
TOAST_NO_BACKUP = "没有可撤销的图像操作"
TOAST_FLATTEN_DONE = "已替换 {n} 张透明底（跳过 {skip}）"
TOAST_FLATTEN_NONE = "范围内没有透明底图片"
TOAST_SCAN_DONE = "找到 {n} 组雷同图片"
TOAST_SCAN_NONE = "没有找到雷同图片"
TOAST_QUARANTINE_DONE = "已将 {n} 张移到 .backups/duplicates/"
TOAST_RESTORE_DONE = "已从备份恢复 {n} 张图片"

OP_FLATTEN = "flatten_alpha"
OP_DUPLICATES = "duplicates"


class ImageToolsBridge(QObject):
    """Runs image-mutating batches off the GUI thread."""

    progress = Signal(str, int, int)  # description, done, total
    flatten_finished = Signal(object)
    groups_ready = Signal(object)
    quarantine_finished = Signal(object)
    restore_finished = Signal(object)

    def __init__(
        self,
        controller: AppController,
        *,
        pool: QThreadPool | None = None,
        parent: QObject | None = None,
    ) -> None:
        super().__init__(parent)
        self._controller = controller
        self._pool = pool if pool is not None else QThreadPool.globalInstance()

    def flatten(self, keys: tuple[str, ...], spec: FlattenSpec) -> bool:
        """Backup then flatten transparent images in ``keys``."""
        paths = self._resolve_paths(keys)
        if paths is None:
            return False
        if not self._controller.begin_work():
            return False
        description = OP_FLATTEN

        def job() -> FlattenReport:
            manager = ImageBackupManager(self._require_root())
            backup = manager.create(OP_FLATTEN, tuple(path for _key, path in paths))
            changed: list[str] = []
            skipped: list[str] = []
            failed: list[tuple[str, str]] = []
            total = len(paths)
            for index, (key, path) in enumerate(paths, start=1):
                try:
                    color = pick_color(spec, key)
                    if flatten_alpha(path, color):
                        changed.append(key)
                    else:
                        skipped.append(key)
                except NLaptError as exc:
                    failed.append((key, exc.message))
                self.progress.emit(description, index, total)
            return FlattenReport(
                changed=tuple(changed),
                skipped=tuple(skipped),
                failed=tuple(failed),
                backup=backup,
            )

        def done(report: object) -> None:
            self._controller.end_work()
            self._controller.refresh()
            if isinstance(report, FlattenReport):
                if report.changed:
                    self._controller.toast_requested.emit(
                        TOAST_FLATTEN_DONE.format(n=len(report.changed), skip=len(report.skipped)),
                        TOAST_OK if not report.failed else TOAST_WARN,
                    )
                else:
                    self._controller.toast_requested.emit(TOAST_FLATTEN_NONE, TOAST_INFO)
            self.flatten_finished.emit(report)

        def failed(message: str) -> None:
            self._controller.end_work()
            self._controller.toast_requested.emit(message, TOAST_ERR)
            self.flatten_finished.emit(None)

        run_async(self._pool, job, on_done=done, on_error=failed)
        return True

    def scan_duplicates(self, keys: tuple[str, ...], max_distance: int) -> bool:
        """Fingerprint ``keys`` and emit similar groups."""
        paths = self._resolve_paths(keys)
        if paths is None:
            return False
        if not self._controller.begin_work():
            return False

        def job() -> tuple[DuplicateGroup, ...]:
            prints = []
            total = len(paths)
            for index, (key, path) in enumerate(paths, start=1):
                prints.append(fingerprint(key, path))
                self.progress.emit(OP_DUPLICATES, index, total)
            return group_similar(tuple(prints), max_distance)

        def done(groups: object) -> None:
            self._controller.end_work()
            count = len(groups) if isinstance(groups, tuple) else 0
            if count:
                self._controller.toast_requested.emit(TOAST_SCAN_DONE.format(n=count), TOAST_OK)
            else:
                self._controller.toast_requested.emit(TOAST_SCAN_NONE, TOAST_INFO)
            self.groups_ready.emit(groups)

        def failed(message: str) -> None:
            self._controller.end_work()
            self._controller.toast_requested.emit(message, TOAST_ERR)
            self.groups_ready.emit(())

        run_async(self._pool, job, on_done=done, on_error=failed)
        return True

    def quarantine(self, paths: tuple[Path, ...]) -> bool:
        """Move unused near-duplicates (and sibling txts) out of the dataset."""
        if not paths:
            self._controller.toast_requested.emit(TOAST_NO_SELECTION, TOAST_WARN)
            return False
        if self._controller.root is None:
            self._controller.toast_requested.emit(TOAST_NO_DATASET, TOAST_WARN)
            return False
        if not self._controller.begin_work():
            return False
        root = self._controller.root

        def job() -> int:
            manager = ImageBackupManager(root)
            info = manager.quarantine(paths)
            return info.file_count

        def done(count: object) -> None:
            self._controller.end_work()
            self._controller.refresh()
            moved = count if isinstance(count, int) else len(paths)
            self._controller.toast_requested.emit(
                TOAST_QUARANTINE_DONE.format(n=moved), TOAST_OK
            )
            self.quarantine_finished.emit(count)

        def failed(message: str) -> None:
            self._controller.end_work()
            self._controller.toast_requested.emit(message, TOAST_ERR)
            self.quarantine_finished.emit(None)

        run_async(self._pool, job, on_done=done, on_error=failed)
        return True

    def restore_last_backup(self) -> bool:
        """Restore the newest flatten backup or quarantine folder."""
        if self._controller.root is None:
            self._controller.toast_requested.emit(TOAST_NO_DATASET, TOAST_WARN)
            return False
        if not self._controller.begin_work():
            return False
        root = self._controller.root

        def job() -> tuple[str, ...]:
            manager = ImageBackupManager(root)
            infos = manager.list_backups()
            if not infos:
                raise NLaptError(TOAST_NO_BACKUP)
            return manager.restore(infos[0])

        def done(keys: object) -> None:
            self._controller.end_work()
            self._controller.refresh()
            restored = keys if isinstance(keys, tuple) else ()
            self._controller.toast_requested.emit(
                TOAST_RESTORE_DONE.format(n=len(restored)), TOAST_OK
            )
            self.restore_finished.emit(keys)

        def failed(message: str) -> None:
            self._controller.end_work()
            kind = TOAST_WARN if message == TOAST_NO_BACKUP else TOAST_ERR
            self._controller.toast_requested.emit(message, kind)
            self.restore_finished.emit(None)

        run_async(self._pool, job, on_done=done, on_error=failed)
        return True

    def _require_root(self) -> Path:
        root = self._controller.root
        if root is None:
            raise NLaptError(TOAST_NO_DATASET)
        return root

    def _resolve_paths(self, keys: tuple[str, ...]) -> list[tuple[str, Path]] | None:
        if self._controller.root is None:
            self._controller.toast_requested.emit(TOAST_NO_DATASET, TOAST_WARN)
            return None
        if not keys:
            self._controller.toast_requested.emit(TOAST_NO_SELECTION, TOAST_WARN)
            return None
        resolved: list[tuple[str, Path]] = []
        for key in keys:
            try:
                resolved.append((key, self._controller.image_path(key)))
            except NLaptError as exc:
                _LOGGER.warning("skip missing image %s: %s", key, exc)
        if not resolved:
            self._controller.toast_requested.emit(TOAST_NO_SELECTION, TOAST_WARN)
            return None
        return resolved
