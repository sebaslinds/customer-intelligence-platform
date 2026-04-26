import logging
import zipfile
from pathlib import Path

from kaggle.api.kaggle_api_extended import KaggleApi

from config.logging_config import configure_logging
from config.settings import get_settings

logger = logging.getLogger(__name__)

DATASET = "yasserh/instacart-online-grocery-basket-analysis-dataset"
OUTPUT_DIR = Path("data/raw")


def download_dataset(dataset: str = DATASET, output_dir: Path = OUTPUT_DIR) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    api = KaggleApi()
    api.authenticate()

    logger.info("Downloading Kaggle dataset %s into %s", dataset, output_dir)
    api.dataset_download_files(dataset, path=output_dir, quiet=False)

    for archive_path in output_dir.glob("*.zip"):
        logger.info("Extracting %s", archive_path)
        with zipfile.ZipFile(archive_path, "r") as archive:
            archive.extractall(output_dir)
        archive_path.unlink()

    logger.info("Dataset download complete")


def main() -> None:
    settings = get_settings()
    configure_logging(settings.log_level)
    download_dataset()


if __name__ == "__main__":
    main()
