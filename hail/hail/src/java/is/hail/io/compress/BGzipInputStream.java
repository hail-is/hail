package is.hail.io.compress;

import htsjdk.samtools.util.BlockCompressedFilePointerUtil;
import is.hail.io.fs.Seekable;
import org.apache.hadoop.io.compress.SplitCompressionInputStream;
import org.apache.hadoop.io.compress.SplittableCompressionCodec;
import java.io.ByteArrayInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.util.zip.GZIPInputStream;
import java.util.zip.ZipException;

public class BGzipInputStream extends SplitCompressionInputStream {
    private static final int BGZF_MAX_BLOCK_SIZE = 64 * 1024;
    private static final int INPUT_BUFFER_CAPACITY = 2 * BGZF_MAX_BLOCK_SIZE;
    private static final int OUTPUT_BUFFER_CAPACITY = BGZF_MAX_BLOCK_SIZE;

    private static final String ZIP_EXCEPTION_MESSAGE = "File does not conform to block gzip format.";

    public static class BGzipHeader {
        /* `bsize' is the size of the current BGZF block.
           It is the `BSIZE' entry of the BGZF extra subfield + 1.  */
        int bsize = 0;

        int isize = 0;

        public int getBlockSize() { return bsize; }

        public BGzipHeader(byte[] buf, int off, int bufSize) throws ZipException {
            if (off + 26 > bufSize)
                throw new ZipException(ZIP_EXCEPTION_MESSAGE);

            if ((buf[off] & 0xff) != 31
                    || (buf[off + 1] & 0xff) != 139
                    || (buf[off + 2] & 0xff) != 8)
                throw new ZipException(ZIP_EXCEPTION_MESSAGE);

            // FEXTRA set
            int flg = (buf[off + 3] & 0xff);
            if ((flg & 4) != 4)
                throw new ZipException(ZIP_EXCEPTION_MESSAGE);

            int xlen = (buf[off + 10] & 0xff) | ((buf[off + 11] & 0xff) << 8);
            if (xlen < 6
                || off + 12 + xlen > bufSize)
                throw new ZipException(ZIP_EXCEPTION_MESSAGE);

            boolean foundBGZFExtraField = false;
            int i = off + 12;
            for (; i < off + 12 + xlen;) {
                if (i + 4 > bufSize)
                    throw new ZipException(ZIP_EXCEPTION_MESSAGE);

                int extraFieldLen = (buf[i + 2] & 0xff) | ((buf[i + 3] & 0xff) << 8);
                if (i + 4 + extraFieldLen > bufSize)
                    throw new ZipException(ZIP_EXCEPTION_MESSAGE);

                if ((buf[i] & 0xff) == 66 && (buf[i + 1] & 0xff) == 67) {
                    if (extraFieldLen != 2)
                        throw new ZipException(ZIP_EXCEPTION_MESSAGE);
                    foundBGZFExtraField = true;
                    bsize = ((buf[i + 4] & 0xff) | ((buf[i + 5] & 0xff) << 8)) + 1;
                }

                i += 4 + extraFieldLen;
            }
            if (i != off + 12 + xlen)
                throw new ZipException(ZIP_EXCEPTION_MESSAGE);
            if (!foundBGZFExtraField
                    || bsize > BGZF_MAX_BLOCK_SIZE)
                throw new ZipException(ZIP_EXCEPTION_MESSAGE);
            if (off + bsize > bufSize)
                throw new ZipException(ZIP_EXCEPTION_MESSAGE);

            isize = ((buf[off + bsize - 4] & 0xff)
                    | ((buf[off + bsize - 3] & 0xff) << 8)
                    | ((buf[off + bsize - 2] & 0xff) << 16)
                    | ((buf[off + bsize - 1] & 0xff) << 24));
            if (isize > BGZF_MAX_BLOCK_SIZE)
                throw new ZipException(ZIP_EXCEPTION_MESSAGE);
        }
    }

    BGzipHeader bgzipHeader;

    final byte[] inputBuffer = new byte[INPUT_BUFFER_CAPACITY];
    int inputBufferSize = 0;
    int inputBufferPos = 0;

    /* `inputBufferInPos' is the position in the compressed input stream corresponding to `inputBuffer[0]'.
     * This position is also the position of the start of the block that is being read from */
    long inputBufferInPos = 0;

    final byte[] outputBuffer = new byte[OUTPUT_BUFFER_CAPACITY];
    int outputBufferSize = 0;
    int outputBufferPos = 0;

    long currentPos;

    public BGzipInputStream(InputStream in, long start, long end, SplittableCompressionCodec.READ_MODE readMode) throws IOException {
        super(in, start, end);

        assert (readMode == SplittableCompressionCodec.READ_MODE.BYBLOCK);
        if (in instanceof org.apache.hadoop.fs.Seekable)
            ((org.apache.hadoop.fs.Seekable) in).seek(start);
        else {
            assert(in instanceof Seekable);
            ((Seekable)in).seek(start);
        }
        resetState();
        decompressNextBlock();

        currentPos = start;
    }

    @Override
    public long getPos() {
        return currentPos;
    }

    public BGzipInputStream(InputStream in) throws IOException {
        this(in, 0L, Long.MAX_VALUE, SplittableCompressionCodec.READ_MODE.BYBLOCK);
    }

    private void fillInputBuffer() throws IOException {
        int newSize = inputBufferSize - inputBufferPos;

        System.arraycopy(inputBuffer, inputBufferPos, inputBuffer, 0, newSize);
        inputBufferInPos += inputBufferPos;
        inputBufferSize = newSize;
        inputBufferPos = 0;

        int needed = inputBuffer.length - inputBufferSize;
        while (needed > 0) {
            int result = in.read(inputBuffer, inputBufferSize, needed);
            if (result < 0)
                break;
            inputBufferSize += result;
            needed = inputBuffer.length - inputBufferSize;
        }
    }

    private void decompressNextBlock() throws IOException {
        outputBufferSize = 0;
        outputBufferPos = 0;

        fillInputBuffer();
        assert (inputBufferPos == 0);
        if (inputBufferSize != 0) {
            bgzipHeader = new BGzipHeader(inputBuffer, inputBufferPos, inputBufferSize);
        } else {
            bgzipHeader = null;
            return;
        }

        int bsize = bgzipHeader.bsize,
                isize = bgzipHeader.isize;

        inputBufferPos += bsize;
        if (isize == 0) {
            decompressNextBlock();
            return;
        }

        InputStream decompIS
                = new GZIPInputStream(new ByteArrayInputStream(inputBuffer, 0, bsize));

        while (outputBufferSize < isize) {
            int result = decompIS.read(outputBuffer, outputBufferSize, isize - outputBufferSize);
            if (result < 0)
                throw new ZipException(ZIP_EXCEPTION_MESSAGE);
            outputBufferSize += result;
        }

        decompIS.close();
    }

    public long blockPos() {
        return inputBufferInPos;
    }

    public long getVirtualOffset() {
        return BlockCompressedFilePointerUtil.makeFilePointer(inputBufferInPos, outputBufferPos);
    }

    public int readBlock(byte[] b) throws IOException {
        if (outputBufferSize == 0)
            return -1;  // EOF
        assert(outputBufferPos == 0);
        assert(outputBufferSize > 0);

        int blockSize = outputBufferSize;
        System.arraycopy(outputBuffer, 0, b, 0, outputBufferSize);

        outputBufferPos = outputBufferSize;
        decompressNextBlock();

        return blockSize;
    }

    public int read(byte[] b, int off, int len) throws IOException {
        if (len == 0)
            return 0;
        if (outputBufferSize == 0)
            return -1;  // EOF
        assert(outputBufferPos < outputBufferSize);

        if (outputBufferPos == 0)
          currentPos = inputBufferInPos + 1;

        int toCopy = Math.min(len, outputBufferSize - outputBufferPos);
        System.arraycopy(outputBuffer, outputBufferPos, b, off, toCopy);
        outputBufferPos += toCopy;

        if (outputBufferPos == outputBufferSize)
            decompressNextBlock();

        return toCopy;
    }

    public int read() throws IOException {
        if (outputBufferSize == 0)
            return -1; // EOF

        if (outputBufferPos == 0)
            currentPos = inputBufferInPos + 1;

        int r = outputBuffer[outputBufferPos];
        outputBufferPos += 1;

        if (outputBufferPos == outputBufferSize)
            decompressNextBlock();

        return r & 0xff;
    }

    public void resetState() throws IOException {
        inputBufferSize = 0;
        inputBufferPos = 0;

        if (in instanceof org.apache.hadoop.fs.Seekable)
            inputBufferInPos = ((org.apache.hadoop.fs.Seekable) in).getPos();
        else {
            assert (in instanceof Seekable);
            inputBufferInPos = ((Seekable) in).getPosition();
        }

        outputBufferSize = 0;
        outputBufferPos = 0;

        // find first block
        fillInputBuffer();

        // the beginning of the file must be a valid bgzip block
        if (inputBufferInPos == 0) {
            new BGzipHeader(inputBuffer, 0, inputBufferSize);
            // inputBufferPos already 0
            return;
        }

        for (int i = 0; i < inputBufferSize - 1; ++i) {
            if ((inputBuffer[i] & 0xff) == 31
                    && (inputBuffer[i + 1] & 0xff) == 139) {
                try {
                    new BGzipHeader(inputBuffer, i, inputBufferSize);
                    inputBufferPos = i;
                    return;
                } catch (ZipException e) {

                }
            }
        }

        assert (inputBufferSize < BGZF_MAX_BLOCK_SIZE);
        inputBufferPos = inputBufferSize;
    }

    // pos is a virtual file pointer, it is not a strict offset into the compressed data.
    // The upper 48 bits of pos are the offset into the compressed data, the lower 16 bits
    // are the offset into the uncompressed block that begins at the pointed to location
    // by the upper 48 bits.
    public void virtualSeek(final long pos) throws IOException {
        final long compOff = BlockCompressedFilePointerUtil.getBlockAddress(pos);
        final int uncompOff = BlockCompressedFilePointerUtil.getBlockOffset(pos);
        if (inputBufferInPos != compOff) {
            if (in instanceof org.apache.hadoop.fs.Seekable)
                ((org.apache.hadoop.fs.Seekable) in).seek(compOff);
            else {
                assert(in instanceof Seekable);
                ((Seekable) in).seek(compOff);
            }

            inputBufferSize = 0;
            inputBufferPos = 0;
            inputBufferInPos = compOff;
            decompressNextBlock();
            assert(inputBufferInPos == compOff);
        }
        if (uncompOff > outputBufferSize || (outputBufferSize > 0 && uncompOff == outputBufferSize)) {
            throw new IOException("Invalid virtual offset: " + pos);
        }
        outputBufferPos = uncompOff;
    }
}
