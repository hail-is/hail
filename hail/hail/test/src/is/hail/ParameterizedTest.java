package is.hail;

import java.lang.annotation.ElementType;
import java.lang.annotation.Retention;
import java.lang.annotation.RetentionPolicy;
import java.lang.annotation.Target;

import org.junit.jupiter.params.provider.ArgumentsSource;

/**
 * Drop-in replacement for the {@code @ParameterizedTest @MethodSource} pair. With no value, the
 * factory method must share a name with the test method (same convention as {@code @MethodSource}).
 * Pass one or more names to use named factory methods shared across tests.
 *
 * <p>Factory methods can return any Scala collection (or Java {@code Iterable}/{@code Iterator}/
 * {@code Stream}/array). Each element is wrapped as one test invocation: Scala tuples are splatted
 * into positional arguments, everything else is passed as a single argument.
 */
@Retention(RetentionPolicy.RUNTIME)
@Target(ElementType.METHOD)
@org.junit.jupiter.params.ParameterizedTest
@ArgumentsSource(HailMethodArgumentsProvider.class)
public @interface ParameterizedTest {
    String value() default "";
}
