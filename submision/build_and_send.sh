python3.7 solve.py --input ~/Downloads/ --output ~/Downloads/result --is-train-set
python3.7 score.py --approx ~/Downloads/result --golden ~/Downloads/train.csv
read -r -p "Are you sure? [press any key] " response
docker build -t vkcup .
docker tag vkcup stor.highloadcup.ru/vkcup21_age/green_barracuda
docker push stor.highloadcup.ru/vkcup21_age/green_barracuda
