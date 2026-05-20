package is.hail;

import java.lang.reflect.Method;
import java.util.Arrays;
import java.util.Iterator;
import java.util.List;
import java.util.NoSuchElementException;
import java.util.Spliterators;
import java.util.function.Predicate;
import java.util.stream.Stream;
import java.util.stream.StreamSupport;

import org.junit.jupiter.api.extension.ExtensionContext;
import org.junit.jupiter.params.provider.Arguments;
import org.junit.jupiter.params.provider.ArgumentsProvider;
import org.junit.jupiter.params.support.AnnotationConsumer;
import org.junit.jupiter.params.support.ParameterDeclarations;
import org.junit.platform.commons.util.CollectionUtils;
import org.junit.platform.commons.util.ReflectionUtils;

/**
 * Backs {@link ParameterizedTest}. Looks up the factory method by name (defaulting to the test
 * method's own name), invokes it through {@link ExtensionContext#getExecutableInvoker()} so its
 * parameters are resolved by the registered {@code ParameterResolver}s (notably {@code
 * HailExtension}), then converts each element into {@link Arguments}. Scala collections and
 * tuples are handled natively.
 */
public class HailMethodArgumentsProvider
    implements ArgumentsProvider, AnnotationConsumer<ParameterizedTest> {

    private String[] sourceNames = new String[0];

    @Override
    public void accept(ParameterizedTest annotation) {
        if (!annotation.value().isEmpty()) {
            this.sourceNames = new String[]{annotation.value()};
        }
    }

    @Override
    public Stream<? extends Arguments> provideArguments(
        ParameterDeclarations parameters, ExtensionContext context) {
        Class<?> testClass = context.getRequiredTestClass();
        Method testMethod = context.getRequiredTestMethod();
        Object testInstance = context.getTestInstance().orElse(null);
        String[] names = sourceNames.length == 0
            ? new String[]{ testMethod.getName() }
            : sourceNames;

        return Arrays.stream(names)
            .map(name -> findFactoryMethod(testClass, name, testMethod))
            .map(m -> context.getExecutableInvoker().invoke(m, testInstance))
            .flatMap(HailMethodArgumentsProvider::toStream)
            .map(HailMethodArgumentsProvider::toArguments);
    }

    private static Method findFactoryMethod(Class<?> cls, String name, Method testMethod) {
        Predicate<Method> isCandidate = m -> m.getName().equals(name) && !m.equals(testMethod);
        List<Method> candidates = ReflectionUtils.findMethods(cls, isCandidate);
        if (candidates.isEmpty()) {
            throw new RuntimeException(
                "no @ParameterizedTest factory method named '" + name + "' on " + cls.getName());
        }
        if (candidates.size() > 1) {
            throw new RuntimeException(
                "ambiguous @ParameterizedTest factory '" + name + "' on " + cls.getName()
                    + ": " + candidates);
        }
        return candidates.get(0);
    }

    private static Stream<?> toStream(Object o) {
        scala.collection.Iterator<?> scalaIt = asScalaIterator(o);
        if (scalaIt != null) {
            return StreamSupport.stream(
                Spliterators.spliteratorUnknownSize(toJavaIterator(scalaIt), 0), false);
        }
        return CollectionUtils.toStream(o);
    }

    private static scala.collection.Iterator<?> asScalaIterator(Object o) {
        if (o instanceof scala.collection.Iterator<?>) {
            return (scala.collection.Iterator<?>) o;
        }
        if (o instanceof scala.collection.Iterable<?>) {
            return ((scala.collection.Iterable<?>) o).iterator();
        }
        return null;
    }

    private static Iterator<Object> toJavaIterator(scala.collection.Iterator<?> sIt) {
        return new Iterator<Object>() {
            @Override public boolean hasNext() { return sIt.hasNext(); }
            @Override public Object next() {
                if (!sIt.hasNext()) throw new NoSuchElementException();
                return sIt.next();
            }
        };
    }

    private static Arguments toArguments(Object o) {
        if (o instanceof Arguments) return (Arguments) o;
        if (o instanceof Object[])  return Arguments.of((Object[]) o);
        if (isScalaTuple(o)) {
            scala.Product p = (scala.Product) o;
            Object[] arr = new Object[p.productArity()];
            for (int i = 0; i < arr.length; i++) {
                arr[i] = p.productElement(i);
            }
            return Arguments.of(arr);
        }
        return Arguments.of(o);
    }

    private static boolean isScalaTuple(Object o) {
        return o instanceof scala.Product
            && o.getClass().getName().startsWith("scala.Tuple");
    }
}
