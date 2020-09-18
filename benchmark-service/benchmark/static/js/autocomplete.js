function auto_complete(place_holder, input_selector) {
     return new autoComplete({
                   data: {
                     src: files,
                     cache: true
                   },
                   sort: (a, b) => {
                       if (a.match < b.match) return -1;
                       if (a.match > b.match) return 1;
                       return 0;
                   },
                   placeHolder: place_holder,
                   selector: input_selector,
                   threshold: 2,
                   debounce: 300,
                   searchEngine: "strict",
                   resultsList: {
                       render: true,
                       container: source => {
                           source.setAttribute("class", "file_list");
                       },
                       destination: document.querySelector(input_selector),
                       position: "afterend",
                       element: "ul"
                   },
                   maxResults: 15,
                   highlight: true,
                   resultItem: {
                       content: (data, source) => {
                           source.innerHTML = data.match;
                       },
                       element: "li"
                   },
                   noResults: () => {
                       const result = document.createElement("li");
                       result.setAttribute("class", "no_result");
                       result.setAttribute("tabindex", "1");
                       result.innerHTML = "No Results";
                       document.querySelector(".autoComplete_list").appendChild(result);
                   },
                   onSelection: feedback => {
                   feedback.event.preventDefault();
                   document.querySelector(input_selector).value = feedback.selection.value;
             }
           });
}
