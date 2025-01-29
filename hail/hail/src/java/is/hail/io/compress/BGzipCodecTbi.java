package is.hail.io.compress;

import htsjdk.samtools.util.FileExtensions;

public class BGzipCodecTbi extends BGzipCodec {
    @Override
    public String getDefaultExtension() {
        return FileExtensions.TABIX_INDEX;
    }
}
