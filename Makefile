all: install generate train run

install:
	pip3 install -r requirements.txt

generate:
	python3 generator/generate_dataset.py

train:
	python3 notebook_source.py

run:
	python3 interfaces/level.py

clean:
	rm -rf ressources/*.pkl
	rm -rf interfaces/public/models.json

fclean: clean
	rm -rf ressources/dataset.csv
