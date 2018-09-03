from pathlib import Path


class Global:
    data_files = str(Path.home()) + '/data'
    external_tools = str(Path.home()) + '/data/imdb'
    STANFORD_PARSER = external_tools + '/parse/stanford-parser-full-2018-02-27/stanford-parser.jar'
    STANFORD_MODELS = external_tools + "/parse/stanford-parser-full-2018-02-27/stanford-parser-3.9.1-models.jar"
