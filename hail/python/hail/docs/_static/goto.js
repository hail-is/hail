$(document).ready(function () {
    var navHeight = $('nav').height();

    if (window.location.hash) {
        var hash = window.location.hash

        var hashName = hash.substring(1, hash.length);
        var elem = $(document.getElementById(hashName));

        if (elem === null) {
            return;
        }

        window.scrollTo(0, parseInt(elem.offset().top, 10) - navHeight);
    }

    $(document).on('click', 'a', function (e) {
        e.preventDefault();
        var hashName = this.href.split('#')[1];

        var elem = $(document.getElementById(hashName));

        if (elem === null) {
            return;
        }

        window.scrollTo(0, parseInt(elem.offset().top, 10) - navHeight);
    })
});