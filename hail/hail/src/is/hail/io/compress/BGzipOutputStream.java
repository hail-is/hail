package is.hail.io.compress;

import org.apache.hadoop.io.compress.CompressionOutputStream;

import java.io.IOException;
import java.io.OutputStream;
import java.util.zip.CRC32;
import java.util.zip.Deflater;

public class BGzipOutputStream extends CompressionOutputStream {

    private final byte[] uncompressedBuffer =
            new byte[BGzipConstants.defaultUncompressedBlockSize];
    private final byte[] compressedBuffer =
            new byte[BGzipConstants.maxCompressedBlockSize - BGzipConstants.blockHeaderLength];
    private final Deflater deflater =
            new Deflater(BGzipConstants.defaultCompressionLevel, true);
    private final Deflater noCompressionDeflater =
            new Deflater(Deflater.NO_COMPRESSION, true);
    private final CRC32 crc32 = new CRC32();

    private boolean finished;
    protected int numUncompressedBytes;


    public BGzipOutputStream(OutputStream out) {
        super(out);
        finished = false;
    }

    @Override
    public void write(int b) throws IOException {
        assert (numUncompressedBytes < uncompressedBuffer.length);
        uncompressedBuffer[numUncompressedBytes] = (byte) b;
        numUncompressedBytes += 1;

        if (numUncompressedBytes == uncompressedBuffer.length) {
            deflateBlock();
        }
    }

    @Override
    public void write(byte[] bytes, int offset, int length) throws IOException {
        assert (numUncompressedBytes < uncompressedBuffer.length);

        int currentPosition = offset;
        int numBytesRemaining = length;

        while (numBytesRemaining > 0) {
            int bytesToWrite =
                    Math.min(uncompressedBuffer.length - numUncompressedBytes, numBytesRemaining);
            System.arraycopy(bytes, currentPosition, uncompressedBuffer, numUncompressedBytes, bytesToWrite);
            numUncompressedBytes += bytesToWrite;
            currentPosition += bytesToWrite;
            numBytesRemaining -= bytesToWrite;
            assert (numBytesRemaining >= 0);

            if (numUncompressedBytes == uncompressedBuffer.length)
                deflateBlock();
        }
    }

    final protected void deflateBlock() throws IOException {
        assert (numUncompressedBytes != 0);
        assert (!finished);

        deflater.reset();
        deflater.setInput(uncompressedBuffer, 0, numUncompressedBytes);
        deflater.finish();
        int compressedSize = deflater.deflate(compressedBuffer, 0, compressedBuffer.length);

        // If it didn't all fit in compressedBuffer.length, set compression level to NO_COMPRESSION
        // and try again.  This should always fit.
        if (!deflater.finished()) {
            noCompressionDeflater.reset();
            noCompressionDeflater.setInput(uncompressedBuffer, 0, numUncompressedBytes);
            noCompressionDeflater.finish();
            compressedSize = noCompressionDeflater.deflate(compressedBuffer, 0, compressedBuffer.length);
            assert (noCompressionDeflater.finished());
        }

        // Data compressed small enough, so write it out.
        crc32.reset();
        crc32.update(uncompressedBuffer, 0, numUncompressedBytes);

        writeGzipBlock(compressedSize, numUncompressedBytes, crc32.getValue());

        numUncompressedBytes = 0; // reset variable
    }

    public void writeInt8(int i) throws IOException {
        out.write(i & 0xff);
    }

    public void writeInt16(int i) throws IOException {
        out.write(i & 0xff);
        out.write((i >> 8) & 0xff);
    }

    public void writeInt32(int i) throws IOException {
        out.write(i & 0xff);
        out.write((i >> 8) & 0xff);
        out.write((i >> 16) & 0xff);
        out.write((i >> 24) & 0xff);
    }

    public int writeGzipBlock(int compressedSize, int bytesToCompress, long crc32val) throws IOException {
        int totalBlockSize = compressedSize + BGzipConstants.blockHeaderLength + BGzipConstants.blockFooterLength;

        writeInt8(BGzipConstants.gzipId1);
        writeInt8(BGzipConstants.gzipId2);
        writeInt8(BGzipConstants.gzipCMDeflate);
        writeInt8(BGzipConstants.gzipFlag);
        writeInt32(BGzipConstants.gzipModificationTime);
        writeInt8(BGzipConstants.gzipXFL);
        writeInt8(BGzipConstants.gzipOsUnknown);
        writeInt16(BGzipConstants.gzipXLEN);
        writeInt8(BGzipConstants.bgzfId1);
        writeInt8(BGzipConstants.bgzfId2);
        writeInt16(BGzipConstants.bgzfLen);
        writeInt16(totalBlockSize - 1);
        out.write(compressedBuffer, 0, compressedSize);
        writeInt32((int) crc32val);
        writeInt32(bytesToCompress);
        return totalBlockSize;
    }

    @Override
    public void resetState() {
        throw new UnsupportedOperationException();
    }

    @Override
    public void finish() throws IOException {
        if (numUncompressedBytes != 0)
            deflateBlock();

        if (!finished) {
            out.write(BGzipConstants.emptyGzipBlock);
            finished = true;
        }
    }
}
