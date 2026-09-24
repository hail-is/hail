package is.hail.io.compress;

public class BGzipCodecGZ extends BGzipCodec {

    @Override
    public String getDefaultExtension() {
        return ".gz";
    }
}
