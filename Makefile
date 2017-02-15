NAME=am-2048
INST_NAME=2048
CC=python3
all: run
run:
	PYTHONUNBUFFERED=1 $(CC) GameManager_3.py
dbuild:
	docker build -t $(NAME) .
drun:  build
	docker run -it --rm --name $(INST_NAME) $(NAME)
