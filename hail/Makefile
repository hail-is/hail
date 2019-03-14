.PHONY: shadowJar test-apiserver

shadowJar:
	./gradlew shadowJar

test-apiserver: shadowJar
	./test-apiserver.sh
