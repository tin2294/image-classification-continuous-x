all: README.md

clean:
	rm README.md

README.md: scenario.md intro.md deploy_app/index.md deploy_k8s/index.md deploy_lb/index.md deploy_hpa/index.md
	pandoc --wrap=none \
		-i scenario.md intro.md \
		deploy_app/index.md \
		deploy_k8s/index.md \
		deploy_lb/index.md \
		deploy_hpa/index.md \
		-o README.md  

