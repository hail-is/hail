package is.hail.io.compress;

import org.apache.hadoop.io.compress.*;

import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;

public class ComposableBGzipCodec implements SplittableCompressionCodec {
    public ComposableBGzipCodec() { }

    @Override
    public Compressor createCompressor() {
        return null;
    }

    @Override
    public Decompressor createDecompressor() {
        return null;
    }

    @Override
    public Class<? extends Compressor> getCompressorType() {
        return null;
    }

    @Override
    public Class<? extends Decompressor> getDecompressorType() {
        return null;
    }

    @Override
    public CompressionInputStream createInputStream(InputStream in) throws IOException {
        return new BGzipInputStream(in);
    }

    @Override
    public CompressionInputStream createInputStream(InputStream in, Decompressor decompressor) throws IOException {
        return createInputStream(in);
    }

    @Override
    public SplitCompressionInputStream createInputStream(InputStream seekableIn,
                                                         Decompressor decompressor, long start, long end, READ_MODE readMode)
            throws IOException {
        return new BGzipInputStream(seekableIn, start, end, readMode);
    }

    @Override
    public CompressionOutputStream createOutputStream(OutputStream out) throws IOException {
        return new ComposableBGzipOutputStream(out);
    }

    @Override
    public CompressionOutputStream createOutputStream(OutputStream out, Compressor compressor) throws IOException {
        return createOutputStream(out);
    }

    @Override
    public String getDefaultExtension() {
        return ".bgz";
    }
}
