document.getElementsByName("table-search-input-box").forEach(input => {
    input.addEventListener("keydown", function (e) {
        if (e.key === "Enter" && (e.metaKey || e.ctrlKey)) {
            e.preventDefault();
            var formId = input.id.substring(0, input.id.lastIndexOf("-input-box")) + "-form";
            document.getElementById(formId).submit();
        }
    })
});
