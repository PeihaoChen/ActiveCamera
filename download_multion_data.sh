mkdir data
cd data
mkdir datasets
cd datasets
gdown "https://drive.google.com/uc?id=14tbqYDWsgC8NXZeikTYvhE3Xo69owuwy"
unzip multinav.zip && rm multinav.zip
cd ../
wget -O objects.zip "http://aspis.cmpt.sfu.ca/projects/multion/objects.zip"
unzip objects.zip && rm objects.zip
wget -O default.phys_scene_config.json "http://aspis.cmpt.sfu.ca/projects/multion/default.phys_scene_config.json"
cd ../
mkdir oracle_maps
cd oracle_maps
gdown "https://drive.google.com/uc?id=1mdII_ksvLxHS-vJ6yhnnRivljaEsqlek"
cd ../
mkdir model_checkpoints
gdown https://drive.google.com/uc?id=19dlIFh0rFf9BeAAjr_5ryo_nk1ds0yI6