from nlpaf.annotator.pipeline.pipeline_base import Pipeline
from nlpaf.util.timer import Timer
from tqdm.auto import tqdm
import pyarrow.parquet as pq
from pathlib import Path
import math
import pandas as pd
from iesta import utils


class IESTAFeatureExtractionPipeline(Pipeline):
    def __init__(
        self,
        input=None,
        load_default_pipe_configs=True,
        extended_pipe_configs: dict = None,
        save_output=False,
        out_path=None,
        argument_col: str = "cleaned_text",
        exec_transformer_based: bool = False,
    ):
        super().__init__(
            input,
            load_default_pipe_configs,
            extended_pipe_configs,
            save_output,
            out_path,
        )
        self.argument_col = argument_col
        self.exec_transformer_based = exec_transformer_based

    def process_input(self) -> list:
        processed = []
        self.input["input_id"] = self.input.index
        txt_df = self.input[["input_id", self.argument_col]].copy()
        txt_df = txt_df.rename(
            columns={self.argument_col: "text"},
        )

        for idx, row in txt_df.iterrows():
            processed.append((row.text, {"input_id": row.input_id}))

        return processed

    def init_and_run(self):
        if self.exec_transformer_based:
            self.add_annotation_pipe(
                name="sentencizer",
                save_output=False,
                is_spacy=True,
                is_native=True,
            )
            self.add_annotation_pipe(
                name="EmotionPipeOrchestrator", save_output=True, is_spacy=True
            )
            self.add_annotation_pipe(
                name="HedgePipeOrchestrator", save_output=True, is_spacy=True
            )
            self.add_annotation_pipe(
                name="ToxicityOrchestrator", save_output=True, is_spacy=True
            )
        else:
            self.add_annotation_pipe(
                name="mpqa_parser",
                save_output=False,
                is_spacy=True,
                is_native=True,
            )
            self.add_annotation_pipe(
                name="MpqaPipeOrchestrator", save_output=True, is_spacy=True
            )
            self.add_annotation_pipe(
                name="EmpathPipeOrchestrator", save_output=True, is_spacy=True
            )

        self.init_pipe_stack()


def extract_features(
    ideology: str,
    data_path: str,
    batch_size: int = 500,
    spacy_n_processors=1,
    transformer_based_features: bool = False,
):
    counter = 1
    data_df = pd.read_parquet(data_path)
    total = math.ceil(len(data_df) / (batch_size))

    parquet_file = pq.ParquetFile(data_path)
    pipeline = IESTAFeatureExtractionPipeline(
        save_output=True, exec_transformer_based=transformer_based_features
    )
    pipeline.spacy_n_processors = (
        1 if transformer_based_features else spacy_n_processors
    )
    pipeline.init_and_run()
    tqdm.pandas()

    customized_feature = (
        "transformer-features"
        if transformer_based_features
        else "style-features"
    )
    out_path = (
        f"data/extracted_features/{ideology}_{customized_feature}_{batch_size}"
    )
    utils.create_folder(out_path)
    out_file = (
        out_path
        + f"/{ideology}_batch"
        + "{}_{}_"
        + customized_feature
        + ".parquet"
    )

    tqdm.pandas()

    for batch in tqdm(
        parquet_file.iter_batches(batch_size=batch_size), total=total
    ):
        file = out_file.format(batch_size, counter, batch_size)

        if Path(file).is_file():
            counter = counter + 1
            continue

        batch_df = batch.to_pandas()

        ## RESET
        pipeline.reset_input_output()
        pipeline.out_path = file

        pipeline.set_input(batch_df)
        # t = Timer(name=f"chunck_{counter}")
        # t.start()

        try:
            pipeline.annotate()
            pipeline.save()
        except:
            print(f"Batch {counter} failed!")
            with open(f"{out_path}/failed.txt", "a") as f:
                f.write(f"{counter}\n")
        # t.stop()
        # pipeline.out_df.head()
        counter = counter + 1


def get_features_df(files, batch_size, training_data) -> pd.DataFrame:
    total = 0
    dfs = []
    for f in files:
        df = pd.read_parquet(f)
        if len(df) != batch_size:
            print(f"{len(df)}: {f}")

        dfs.append(df)
    features_df: pd.DataFrame = pd.concat(dfs)
    features_df.set_index("input_id", inplace=True)

    df_ = pd.merge(
        training_data, features_df, left_index=True, right_index=True
    )
    assert len(df_) == len(features_df)

    return df_
