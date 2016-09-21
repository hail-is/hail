package org.broadinstitute.hail.io.compress;

public class BGzipCodecGZ extends BGzipCodec {

    @Override
    public String getDefaultExtension() {
        return ".gz";
    }
}
