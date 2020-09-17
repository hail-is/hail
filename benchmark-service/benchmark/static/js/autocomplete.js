
    new autoComplete({
        data: {                              // Data src [Array, Function, Async] | (REQUIRED)
          src: files,
          cache: true
        },
        sort: (a, b) => {                    // Sort rendered results ascendingly | (Optional)
            if (a.match < b.match) return -1;
            if (a.match > b.match) return 1;
            return 0;
        },
        placeHolder: "gs://...",     // Place Holder text                 | (Optional)
        selector: ".autoComplete",           // Input field selector              | (Optional)
        threshold: 2,                        // Min. Chars length to start Engine | (Optional)
        debounce: 300,                       // Post duration for engine to start | (Optional)
        searchEngine: "strict",              // Search Engine type/mode           | (Optional)
        resultsList: {                       // Rendered results list object      | (Optional)
            render: true,
            /* if set to false, add an eventListener to the selector for event type
               "autoComplete" to handle the result */
            container: source => {
                source.setAttribute("class", "file_list");
            },
            destination: document.querySelector(".autoComplete"),
            position: "afterend",
            element: "ul"
        },
        maxResults: 15,                         // Max. number of rendered results | (Optional)
        highlight: true,                       // Highlight matching results      | (Optional)
        resultItem: {                          // Rendered result item            | (Optional)
            content: (data, source) => {
                source.innerHTML = data.match;
            },
            element: "li"
        },
        noResults: () => {                     // Action script on noResults      | (Optional)
            const result = document.createElement("li");
            result.setAttribute("class", "no_result");
            result.setAttribute("tabindex", "1");
            result.innerHTML = "No Results";
            document.querySelector(".autoComplete_list").appendChild(result);
        },
        onSelection: feedback => {
        feedback.event.preventDefault();
        document.querySelector(".autoComplete").value = feedback.selection.value;
  }
});


new autoComplete({
        data: {                              // Data src [Array, Function, Async] | (REQUIRED)
          src: files,
          cache: true
        },
        sort: (a, b) => {                    // Sort rendered results ascendingly | (Optional)
            if (a.match < b.match) return -1;
            if (a.match > b.match) return 1;
            return 0;
        },
        placeHolder: "gs://...",     // Place Holder text                 | (Optional)
        selector: "#autoComplete2",           // Input field selector              | (Optional)
        threshold: 2,                        // Min. Chars length to start Engine | (Optional)
        debounce: 300,                       // Post duration for engine to start | (Optional)
        searchEngine: "strict",              // Search Engine type/mode           | (Optional)
        resultsList: {                       // Rendered results list object      | (Optional)
            render: true,
            /* if set to false, add an eventListener to the selector for event type
               "autoComplete" to handle the result */
            container: source => {
                source.setAttribute("class", "file_list");
            },
            destination: document.querySelector("#autoComplete2"),
            position: "afterend",
            element: "ul"
        },
        maxResults: 15,                         // Max. number of rendered results | (Optional)
        highlight: true,                       // Highlight matching results      | (Optional)
        resultItem: {                          // Rendered result item            | (Optional)
            content: (data, source) => {
                source.innerHTML = data.match;
            },
            element: "li"
        },
        noResults: () => {                     // Action script on noResults      | (Optional)
            const result = document.createElement("li");
            result.setAttribute("class", "no_result");
            result.setAttribute("tabindex", "1");
            result.innerHTML = "No Results";
            document.querySelector(".autoComplete_list").appendChild(result);
        },
        onSelection: feedback => {
        feedback.event.preventDefault();
        document.querySelector("#autoComplete2").value = feedback.selection.value;
  }
});
