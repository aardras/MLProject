from ingestion import Ingest
from preprocessing import Preprocessor
from splitter import Split, SplitClass
from dataclasses import dataclass

@dataclass
class PipelineResult:
    schema: dict
    split: Split

    def summary(self):
        report=[
            "Pipeline Summary",
            f"File: {self.schema['filepath']}",
            f"Colnames: {self.schema['colnames']}",
            f"Data types: {self.schema['datatypes']}",
            f"Null counts: {self.schema['null counts']}",
            f"Total entries: {self.schema['total entries']}"
        ]

        for name, shape in self.split.shapes().items():
            report.append(f"{name}:{shape}")
        return "\n".join(report)

class DataPipeline:
    def __init__(self, config:dict):
        self.config = config or {}
        self.ingestor = Ingest()
        self.preprocessor = Preprocessor(config=self.config)
        self.splitter = SplitClass(config=self.config)
    
    def run(self, filepath: str):
        schema = self.ingestor.ingest(filepath)
        df = schema['data']

        df_processed = self.preprocessor.fit_transform(df)

        split = self.splitter.split(df_processed)

        return PipelineResult(schema = schema, split = split)
