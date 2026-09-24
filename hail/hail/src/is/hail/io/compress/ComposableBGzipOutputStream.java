package is.hail.io.compress;

import java.io.IOException;
import java.io.OutputStream;

public final class ComposableBGzipOutputStream extends BGzipOutputStream {

    public ComposableBGzipOutputStream(OutputStream out) {
        super(out);
    }

    @Override
    public void finish() throws IOException {
        if (numUncompressedBytes != 0) {
            deflateBlock();
        }
    }
}
