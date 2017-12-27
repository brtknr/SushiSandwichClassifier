# Sushi or Sandwich classifier

WORKDIR=~/Coding/SushiSandwichClassifier

	cd $WORKDIR
	docker build -t sushi .
	docker run -it -p 9999:9999 -v $WORKDIR/:/opt/sushi/ sushi jupyter notebook --port 9999 --ip=0.0.0.0 --allow-root --no-browser --notebook-dir=/opt/sushi/