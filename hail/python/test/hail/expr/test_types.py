import unittest
from typing import Optional

import hail as hl
from hail.expr import coercer_from_dtype, dtype
from hail.utils.java import Env

from ..helpers import fails_local_backend, resource, skip_unless_spark_backend, skip_when_service_backend


class Tests(unittest.TestCase):
    def types_to_test(self):
        return [
            hl.tint32,
            hl.tint64,
            hl.tfloat32,
            hl.tfloat64,
            hl.tstr,
            hl.tbool,
            hl.tcall,
            hl.tinterval(hl.tint32),
            hl.tdict(hl.tstr, hl.tint32),
            hl.tarray(hl.tstr),
            hl.tndarray(hl.tstr, 1),
            hl.tndarray(hl.tfloat64, 2),
            hl.tset(hl.tint64),
            hl.tlocus('GRCh37'),
            hl.tlocus('GRCh38'),
            hl.tstruct(),
            hl.tstruct(x=hl.tint32, y=hl.tint64, z=hl.tarray(hl.tset(hl.tstr))),
            hl.tstruct(**{
                'weird field name 1': hl.tint32,
                r"""this one ' has "" quotes and `` backticks```""": hl.tint64,
                '!@#$%^&({[': hl.tarray(hl.tset(hl.tstr)),
            }),
            hl.tinterval(hl.tlocus()),
            hl.tset(hl.tinterval(hl.tlocus())),
            hl.tstruct(a=hl.tint32, b=hl.tint32, c=hl.tarray(hl.tstr)),
            hl.tstruct(a=hl.tfloat64, bb=hl.tint32, c=hl.tbool),
            hl.tstruct(a=hl.tint32, b=hl.tint32),
            hl.tstruct(**{'___': hl.tint32, '_ . _': hl.tint32}),
            hl.tunion(),
            hl.tunion(a=hl.tint32, b=hl.tstr),
            hl.tunion(**{'!@#$%^&({[': hl.tstr}),
            hl.ttuple(hl.tstr, hl.tint32),
            hl.ttuple(hl.tarray(hl.tint32), hl.tstr, hl.tstr, hl.tint32, hl.tbool),
            hl.ttuple(),
        ]

    def test_parser_roundtrip(self):
        for t in self.types_to_test():
            self.assertEqual(t, dtype(str(t)))

    def test_eval_roundtrip(self):
        for t in self.types_to_test():
            self.assertEqual(t, eval(repr(t)))

    def test_equality(self):
        ts = self.types_to_test()
        ts2 = self.types_to_test()  # reallocates the non-primitive types

        for i in range(len(ts)):
            for j in range(len(ts2)):
                if i == j:
                    self.assertEqual(ts[i], ts2[j])
                else:
                    self.assertNotEqual(ts[i], ts2[j])

    @skip_unless_spark_backend()
    def test_type_jvm_roundtrip(self):
        ts = self.types_to_test()
        for t in ts:
            rev_str = t._parsable_string()
            jtyp = Env.hail().expr.ir.IRParser.parseType(rev_str)
            self.assertEqual(t, dtype(jtyp.toString()))

    def test_pretty_roundtrip(self):
        ts = self.types_to_test()
        for t in ts:
            p1 = t.pretty()
            p2 = t.pretty(5, 5)
            self.assertEqual(t, dtype(p1))
            self.assertEqual(t, dtype(p2))

    def test_coercers_can_coerce(self):
        ts = self.types_to_test()
        for t in ts:
            c = coercer_from_dtype(t)
            self.assertTrue(c.can_coerce(t))
            self.assertFalse(c.requires_conversion(t))

    @skip_when_service_backend(reason='to_spark is nonsensical in the service')
    @fails_local_backend()
    def test_nested_type_to_spark(self):
        ht = hl.utils.range_table(10)
        ht = ht.annotate(nested=hl.dict({"tup": hl.tuple([ht.idx])}))
        ht.to_spark()  # should not throw exception

    def test_rename_not_unique(self):
        with self.assertRaisesRegex(ValueError, "attempted to rename 'b' and 'c' both to 'x'"):
            hl.tstruct(a=hl.tbool, b=hl.tint32, c=hl.tint32)._rename({'b': 'x', 'c': 'x'})
        with self.assertRaisesRegex(ValueError, "attempted to rename 'a' and 'b' both to 'a'"):
            hl.tstruct(a=hl.tbool, b=hl.tint32)._rename({'b': 'a'})

    def test_get_context(self):
        tl1 = hl.tlocus('GRCh37')
        tl2 = hl.tlocus('GRCh38')

        types_and_rgs = [
            (
                [
                    hl.tint32,
                    hl.tint64,
                    hl.tfloat32,
                    hl.tfloat64,
                    hl.tstr,
                    hl.tbool,
                    hl.tcall,
                    hl.tinterval(hl.tset(hl.tint32)),
                    hl.tdict(hl.tstr, hl.tarray(hl.tint32)),
                    hl.tndarray(hl.tstr, 1),
                    hl.tstruct(),
                    hl.tstruct(x=hl.tint32, y=hl.tint64, z=hl.tarray(hl.tset(hl.tstr))),
                    hl.tunion(),
                    hl.tunion(a=hl.tint32, b=hl.tstr),
                    hl.ttuple(hl.tstr, hl.tint32),
                    hl.ttuple(),
                ],
                set(),
            ),
            (
                [
                    tl1,
                    hl.tinterval(tl1),
                    hl.tdict(hl.tstr, tl1),
                    hl.tndarray(tl1, 2),
                    hl.tinterval(tl1),
                    hl.tset(hl.tinterval(tl1)),
                    hl.tstruct(a=hl.tint32, b=hl.tint32, c=hl.tarray(tl1)),
                    hl.tunion(a=hl.tint32, b=tl1),
                    hl.ttuple(hl.tarray(hl.tint32), tl1, hl.tstr, hl.tint32, hl.tbool),
                ],
                {"GRCh37"},
            ),
            (
                [
                    hl.tdict(tl1, tl2),
                    hl.ttuple(hl.tarray(tl2), tl1, hl.tstr, hl.tint32, hl.tbool),
                ],
                {"GRCh37", "GRCh38"},
            ),
        ]

        for types, rgs in types_and_rgs:
            for t in types:
                self.assertEqual(t.get_context().references, rgs)

    def test_tlocus_schema_from_rg_matches_scala(self):
        def locus_from_import_vcf(rg: Optional[str]) -> hl.HailType:
            return hl.import_vcf(resource('sample2.vcf'), reference_genome=rg).locus.dtype

        assert hl.tlocus._schema_from_rg(None) == locus_from_import_vcf(None)
        assert hl.tlocus._schema_from_rg('GRCh37') == locus_from_import_vcf('GRCh37')
        assert hl.tlocus._schema_from_rg('GRCh38') == locus_from_import_vcf('GRCh38')
        assert hl.tlocus._schema_from_rg('GRCm38') == locus_from_import_vcf('GRCm38')
        assert hl.tlocus._schema_from_rg('CanFam3') == locus_from_import_vcf('CanFam3')
        assert hl.tlocus._schema_from_rg('default') == locus_from_import_vcf('default')
