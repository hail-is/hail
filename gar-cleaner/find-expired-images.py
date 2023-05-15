import hail as hl
import sys

NAMESPACE = sys.argv[0]
KEEP_THIS_MANY_IMAGES_PER_NAME_PER_KIND = 10

ht = hl.import_table('images5', delimiter='\t')
ht = ht.group_by(ht.IMAGE).aggregate(
    images=hl.agg.collect(hl.struct(digest=ht.DIGEST, tags=ht.TAGS, create=ht.CREATE_TIME, update=ht.UPDATE_TIME))
)
kinds = ['pr', 'dev', 'deploy']
ht = ht.annotate(kind=[*kinds, hl.missing(hl.tstr)])
ht = ht.explode('kind')
ht = ht.annotate(
    images=ht.images.filter(
        lambda x: hl.any(
            hl.if_else(
                hl.is_defined(ht.kind),
                x.tags[: hl.len(ht.kind)] == ht.kind,
                hl.all(*[x.tags[: hl.len(k)] != (k) for k in kinds]),
            )
        )
    )
)
ht = ht.annotate(images=hl.sorted(ht.images, lambda x: (x.update, x.create)))
ht = ht.transmute(
    keepers=ht.images[-KEEP_THIS_MANY_IMAGES_PER_NAME_PER_KIND:],
    trash=ht.images[:-KEEP_THIS_MANY_IMAGES_PER_NAME_PER_KIND],
)
ht = ht.checkpoint('tmp.ht', overwrite=True)
ht.show(100)

keepers = ht
keepers = keepers.select('keepers').explode('keepers')
keepers = keepers.transmute(**keepers.keepers)
keepers = keepers.key_by(name=keepers.IMAGE + "@" + keepers.digest)
keepers = keepers.checkpoint('images-with-expired-tags.ht', overwrite=True)
keepers.show(100)

trash = ht
trash = trash.select('trash').explode('trash')
trash = trash.transmute(**trash.trash)
trash = trash.key_by(name=trash.IMAGE + "@" + trash.digest)
if NAMESPACE != 'default':
    trash = trash.filter(trash.tags.contains(NAMESPACE))
trash = trash.checkpoint('images-with-expired-tags.ht', overwrite=True)
trash.show(100)

expired = hl.read_table('images-with-expired-tags.ht')
expired = expired.filter(hl.is_missing(keepers[expired.name]))
expired = expired.select()
expired.export('expired-images.csv', header=False)
