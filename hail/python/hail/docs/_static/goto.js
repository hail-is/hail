//https://caniuse.com/#feat=history
if ((window.history && window.history.pushState)) {
    var startingHash = window.location.hash ? window.location.hash.replace('#', '') : null;

    history.pushState("", document.title, window.location.pathname);

    $(document).ready(function () {
        var navHeight = $('nav').height();

        if (startingHash) {
            var elem = document.getElementById(startingHash);

            if (!elem) {
                return;
            }

            window.scrollTo(0, parseInt($(elem).offset().top, 10) - navHeight);
            history.pushState({}, null, `#${startingHash}`);
        }


        $(document).on('click', 'a', function (e) {
            var hrefParts = this.href.split('#');
            var hash = hrefParts[1];

            var elem = document.getElementById(hash);

            if (!elem) {
                return;
            }

            e.preventDefault();
            e.stopPropagation();
            window.scrollTo(0, parseInt($(elem).offset().top, 10) - navHeight);
            history.pushState({}, null, `#${hash}`);
        })
    })
} else {
    console.warn("Histroy API unsupported. Please consider updating your browser");
}