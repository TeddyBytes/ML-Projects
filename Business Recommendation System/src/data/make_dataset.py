import click
import logging
from pathlib import Path
from dotenv import find_dotenv, load_dotenv
import pandas as pd
from datetime import datetime
import os

# Import all the functions we defined earlier
from your_module import (
    process_business_data,
    process_user_data,
    process_review_data,
    filter_and_merge_data,
    normalize_data,
    tokenize_and_process,
)


@click.command()
@click.argument("input_filepath", type=click.Path(exists=True))
@click.argument("output_filepath", type=click.Path())
def main(input_filepath, output_filepath):
    """Runs data processing scripts to turn raw data from (../raw) into
    cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info("making final data set from raw data")

    # Load raw data
    data_businesses = pd.read_parquet(
        os.path.join(local_path, "yelp_academic_dataset_business.parquet")
    )
    data_review = pd.read_parquet(
        os.path.join(local_path, "yelp_academic_dataset_review.parquet")
    )
    data_user = pd.read_parquet(
        os.path.join(local_path, "yelp_academic_dataset_user.parquet")
    )

    current_time = datetime.now()

    # Define categorical features and target
    categorical_features = [
        "region_code",
        "state_code",
        "city_code",
        "categories_enc",
        "day_of_year",
        "day_of_week",
    ]
    target = ["mean_centered_rating"]

    # Process data
    data_businesses = process_business_data(data_businesses)
    data_user = process_user_data(data_user, current_time)
    data_review = process_review_data(data_review, current_time)

    data_review_subset = data_review[
        [
            "user_id",
            "business_id",
            "rating",
            "clean_text",
            "years_since_review",
            "day_of_week",
            "day_of_year",
        ]
    ]
    data_user_subset = data_user[
        ["years_yelp_member", "user_id", "user_review_count", "user_avg_rating"]
    ]
    data_businesses_subset = data_businesses[
        [
            "business_id",
            "bus_avg_rating",
            "business_review_count",
            "region_code",
            "state_code",
            "city_code",
            "categories_enc",
        ]
    ]

    pre_norm_df = filter_and_merge_data(
        data_review_subset, data_user_subset, data_businesses_subset
    )
    pre_token_df = normalize_data(
        pre_norm_df, categorical_features, target, output_filepath
    )
    final_df = tokenize_and_process(pre_token_df, local_path=output_filepath)

    logger.info("Final dataset created and saved")


if __name__ == "__main__":
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
