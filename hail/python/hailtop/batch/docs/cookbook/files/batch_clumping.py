import hailtop.batch as hb


def gwas(batch, vcf, phenotypes):
    """
    QC data and get association test statistics
    """
    cores = 2
    g = batch.new_job(name='run-gwas')
    g.image('us-docker.pkg.dev/<MY_PROJECT>/1kg-gwas:latest')
    g.cpu(cores)
    g.declare_resource_group(
        ofile={'bed': '{root}.bed', 'bim': '{root}.bim', 'fam': '{root}.fam', 'assoc': '{root}.assoc'}
    )
    g.command(
        f"""
python3 /run_gwas.py \
    --vcf {vcf} \
    --phenotypes {phenotypes} \
    --output-file {g.ofile} \
    --cores {cores}
"""
    )
    return g


def clump(batch, bfile, assoc, chr):
    """
    Clump association results with PLINK
    """
    c = batch.new_job(name=f'clump-{chr}')
    c.image('hailgenetics/genetics:0.2.37')
    c.memory('1Gi')
    c.command(
        f"""
plink --bfile {bfile} \
    --clump {assoc} \
    --chr {chr} \
    --clump-p1 0.01 \
    --clump-p2 0.01 \
    --clump-r2 0.5 \
    --clump-kb 1000 \
    --memory 1024

mv plink.clumped {c.clumped}
"""
    )
    return c


def merge(batch, results):
    """
    Merge clumped results files together
    """
    merger = batch.new_job(name='merge-results')
    merger.image('ubuntu:22.04')
    if results:
        merger.command(
            f"""
head -n 1 {results[0]} > {merger.ofile}
for result in {" ".join(results)}
do
    tail -n +2 "$result" >> {merger.ofile}
done
sed -i -e '/^$/d' {merger.ofile}
"""
        )
    return merger


if __name__ == '__main__':
    backend = hb.ServiceBackend()
    batch = hb.Batch(backend=backend, name='clumping')

    vcf = batch.read_input('gs://hail-tutorial/1kg.vcf.bgz')
    phenotypes = batch.read_input('gs://hail-tutorial/1kg_annotations.txt')

    g = gwas(batch, vcf, phenotypes)

    results = []
    for chr in range(1, 23):
        c = clump(batch, g.ofile, g.ofile.assoc, chr)
        results.append(c.clumped)

    m = merge(batch, results)
    batch.write_output(m.ofile, 'gs://<MY_BUCKET>/batch-clumping/1kg-caffeine-consumption.clumped')

    batch.run(open=True, wait=False)
    backend.close()
