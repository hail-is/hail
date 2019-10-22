//https://caniuse.com/#feat=history
if ((window.history && window.history.pushState)) {
    var startingHash = window.location.hash ? decodeURIComponent(window.location.hash.replace('#', '')) : null;

    // necessary to prevent browser from overriding our initial scroll
    // browser scroll is otherwise undefeatable
    history.pushState("", document.title, window.location.pathname);
    // necessary for chrome
    window.history.scrollRestoration = "manual";

    $(document).ready(function () {
        MathJax.Hub.Register.StartupHook("End", function () {
            var navHeight = $('nav').height();

            if (startingHash) {
                var elem = document.getElementById(startingHash);

                if (!elem) {
                    return;
                }

                // setTimeout is necessary for safari, but not firefox or chrome
                setTimeout(() => {
                    window.scrollTo(0, parseInt($(elem).offset().top, 10) - navHeight);
                    history.pushState({}, null, `#${startingHash}`);
                }, 0)
            }

            $(document).on('click', 'a', function (e) {
                var currentHref = location.href.split("#")[0];
                var hrefParts = this.href.split('#');

                if (hrefParts.length == 1 || hrefParts[0] !== currentHref) {
                    return;
                }

                var hash = decodeURIComponent(hrefParts[1]);
                var elem = document.getElementById(hash);

                if (!elem) {
                    console.warn(`Couldn't find element with id ${hash}`)
                    return;
                }

                e.preventDefault();
                window.scrollTo(0, parseInt($(elem).offset().top, 10) - navHeight);
                history.pushState({}, null, `#${hash}`);
            });
        });
    });
} else {
    console.warn("Histroy API unsupported. Please consider updating your browser");
}