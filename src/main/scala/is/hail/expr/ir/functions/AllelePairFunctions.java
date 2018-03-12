package is.hail.expr.ir.functions;

import static java.lang.Math.sqrt;

public class AllelePairFunctions {

    private static int[] cachedAllelePairs = new int[]{
            allelePair(0, 0),
            allelePair(0, 1), allelePair(1, 1),
            allelePair(0, 2), allelePair(1, 2), allelePair(2, 2),
            allelePair(0, 3), allelePair(1, 3), allelePair(2, 3), allelePair(3, 3),
            allelePair(0, 4), allelePair(1, 4), allelePair(2, 4), allelePair(3, 4), allelePair(4, 4),
            allelePair(0, 5), allelePair(1, 5), allelePair(2, 5), allelePair(3, 5), allelePair(4, 5), allelePair(5, 5),
            allelePair(0, 6), allelePair(1, 6), allelePair(2, 6), allelePair(3, 6), allelePair(4, 6), allelePair(5, 6), allelePair(6, 6),
            allelePair(0, 7), allelePair(1, 7), allelePair(2, 7), allelePair(3, 7), allelePair(4, 7), allelePair(5, 7), allelePair(6, 7), allelePair(7, 7)
    };

    public static int allelePair(int j, int k) {
        return j | (k << 16);
    }

    public static int calculateAllelePairFromRepr(int repr) {
        int k = (int) (sqrt(8 * (double)repr + 1)/2 - 0.5);
        int j = repr - k * (k + 1) / 2;
        return allelePair(j, k);
    }

    public static int allelePair(int repr) {
        if (repr < cachedAllelePairs.length) {
            return cachedAllelePairs[repr];
        } else {
            return calculateAllelePairFromRepr(repr);
        }
    }

    public static int j(int pair) {
        return pair & 0xffff;
    }

    public static int k(int pair) {
        return (pair >> 16) & 0xffff;
    }

    public static int allelePairFromPhased(int repr) {
        int p = allelePair(repr);
        return allelePair(j(p), k(p) - j(p));
    }
}
