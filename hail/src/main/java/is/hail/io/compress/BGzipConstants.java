package is.hail.io.compress;

public interface BGzipConstants {

    /**
     * Number of bytes in the gzip block before the deflated data.
     */
    int blockHeaderLength = 18;

    /**
     * Location in the gzip block of the total block size (actually total block size - 1)
     */
    int blockLengthOffset = 16;

    /**
     * Number of bytes that follow the deflated data
     */
    int blockFooterLength = 8;

    /**
     * We require that a compressed block (including header and footer, be <= this)
     */
    int maxCompressedBlockSize = 64 * 1024;

    /**
     * Gzip overhead is the header, the footer, and the block size (encoded as a short).
     */
    int gzipOverhead = blockHeaderLength + blockFooterLength + 2;

    /**
     * If Deflater has compression level == NO_COMPRESSION, 10 bytes of overhead (determined experimentally).
     */
    int noCompressionOverhead = 10;

    /**
     * Push out a gzip block when this many uncompressed bytes have been accumulated.
     */
    int defaultUncompressedBlockSize = 64 * 1024 - (gzipOverhead + noCompressionOverhead);

    // gzip magic numbers

    int gzipId1 = 31;
    int gzipId2 = 139;

    int gzipModificationTime = 0;

    /**
     * set extra fields to true
     */
    int gzipFlag = 4;

    /**
     * extra flags
     */
    int gzipXFL = 0;

    /**
     * length of extra subfield
     */
    int gzipXLEN = 6;

    /**
     * The deflate compression, which is customarily used by gzip
     */
    int gzipCMDeflate = 8;
    int defaultCompressionLevel = 5;
    int gzipOsUnknown = 255;
    int bgzfId1 = 66;
    int bgzfId2 = 67;
    int bgzfLen = 2;

    byte[] emptyGzipBlock = new byte[]{
            0x1f, (byte) 0x8b, 0x08, 0x04, 0x00, 0x00, 0x00, 0x00,
            0x00, (byte) 0xff, 0x06, 0x00, 0x42, 0x43, 0x02, 0x00,
            0x1b, 0x00, 0x03, 0x00, 0x00, 0x00, 0x00, 0x00,
            0x00, 0x00, 0x00, 0x00
    };
}