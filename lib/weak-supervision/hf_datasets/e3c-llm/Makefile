docker-build:
	docker build -t e3c-llm .

docker-run:
	docker run -v $(PWD):/workspace/project -it -e OPENAI_API_KEY=$(OPENAI_API_KEY) -e shell=/bin/zsh --rm e3c-llm zsh
