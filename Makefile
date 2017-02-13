NAME=am-2048
INST_NAME=2048
CC=pypy3
all: run
run:
	$(CC) GameManager_3.py
dbuild:
	docker build -t $(NAME) .
drun:  build
	docker run -it --rm --name $(INST_NAME) $(NAME)
