NAME=am-2048
INST_NAME=2048
all: run
build:
	docker build -t $(NAME) .
run:  build
	docker run -it --rm --name $(INST_NAME) $(NAME)
