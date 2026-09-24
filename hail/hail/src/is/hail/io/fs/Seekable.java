package is.hail.io.fs;

public interface Seekable extends Positioned {
    void seek(long position);
}
