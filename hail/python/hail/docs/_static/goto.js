//https://caniuse.com/#feat=history
if ((window.history && window.history.pushState && window.scrollTo)) {
    $(document).ready(function () {
        var navHeight = $('nav').height();
        var hash = window.location.hash ? decodeURIComponent(window.location.hash.replace('#', '')) : null;

        if (hash) {
            var elem = document.getElementById(hash);

            if (!elem) {
                return;
            }

            setTimeout(() => {
                window.scrollTo(0, parseInt($(elem).offset().top, 10) - navHeight);
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
            setTimeout(() => {
                window.scrollTo(0, parseInt($(elem).offset().top, 10) - navHeight);
                history.pushState({}, null, `#${hash}`);
            }, 0);
        });
    });
} else {
    console.warn("Histroy API or scrollTo unsupported. Please consider updating your browser");
}