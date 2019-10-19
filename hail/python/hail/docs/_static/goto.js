//https://caniuse.com/#feat=history
if ((window.history && window.history.pushState)) {
    var startingHash = window.location.hash,
        startingHash = startingHash.replace('#', '');

    history.pushState("", document.title, window.location.pathname);

    $(document).ready(function () {
        var navHeight = $('nav').height();

        if (startingHash) {
            var hashName = startingHash;
            var elem = document.getElementById(hashName);

            if (!elem) {
                return;
            }

            window.scrollTo(0, parseInt($(elem).offset().top, 10) - navHeight);
            history.pushState({}, null, `#${hashName}`);
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