.PHONY: shadowJar jars clean

shadowJar:
	./gradlew shadowJar

jars:
	make -C src/main/c
	mvn -B compile test-compile jar:jar jar:test-jar assembly:single

clean:
	make -C src/main/c clean
	mvn -B clean
