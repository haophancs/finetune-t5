import os

os.system('wget -P ./data/ https://downloads.wortschatz-leipzig.de/corpora/vie_wikipedia_2021_1M.tar.gz')
os.system('tar -xvzf ./data/vie_wikipedia_2021_1M.tar.gz -C ./data/')