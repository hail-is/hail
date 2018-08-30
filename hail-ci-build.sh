set -x
source activate hail
GRADLE_OPTS=-Xmx2048m ./gradlew testAll makeDocs archiveZip --gradle-user-home /gradle-cache && \
    pip install -U cloudtools && \
    cluster start ci-test-$SOURCE_SHA-$TARGET_SHA \
            --version devel \
            --spark 2.2.0 \
            --jar build/libs/hail-all-spark.jar \
            --zip build/distributions/hail-python.zip && \
    cluster submit ci-test-$SOURCE_SHA-$TARGET_SHA \
            cluster-sanity-check.py
EXIT_CODE=$?
rm -rf artifacts
mkdir -p artifacts
cp build/libs/hail-all-spark.jar artifacts/hail-all-spark.jar
cp build/distributions/hail-python.zip artifacts/hail-python.zip
cp -R build/www artifacts/www
cp -R build/reports/tests artifacts/test-report
cat <<EOF > artifacts/index.html
<html>
<body>
<h1>$(git rev-parse HEAD)</h1>
<ul>
<li><a href='hail-all-spark.jar'>hail-all-spark.jar</a></li>
<li><a href='hail-python.zip'>hail-python.zip</a></li>
<li><a href='www/index.html'>www/index.html</a></li>
<li><a href='test-report/index.html'>test-report/index.html</a></li>
</ul>
</body>
</html>
EOF
exit $EXIT_CODE
