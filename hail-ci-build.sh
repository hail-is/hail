set -ex
source activate hail
GRADLE_OPTS=-Xmx2048m ./gradlew testAll --gradle-user-home /gradle-cache
