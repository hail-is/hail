from hailtop import pipeline

p = pipeline.Pipeline(
    backend=pipeline.GoogleBackend(
        scratch_dir='gs://hail-cseed/cs-hack/tmp',
        worker_cores=1,
        worker_disk_size_gb='20',
        pool_size=3,
        max_instances=1000),
    default_image='ubuntu:18.04')

input = p.read_input('gs://hail-cseed/cs-hack/input.txt')

t1 = p.new_task('concat')
t1.command(f'cp {input} {t1.ofile} && echo "end" >> {t1.ofile}')

t2 = p.new_task('sum')
t2.command(f'sum {t1.ofile} > {t2.sum}')

p.write_output(t2.sum, 'gs://hail-cseed/cs-hack/sum.txt')

p.run()
