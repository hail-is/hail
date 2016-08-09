# Adding documentation for commands in Hail

0. Make sure you add enough information to the args4j options and the `Command.description` method as these will be used to auto-generate content. In addition, add a `metaVar` parameter to the args4j option. Example: `metaVar = "MAF"` in ImputeSex.scala.

1. Make a new file in the `docs/commands/` directory with the same name as the command `Command.name` with all spaces or slashes replaced by underscores.
    
    Example: `annotateglobal table` becomes `annotateglobal_table`
 
 
2. Put the following statement at the top of the file which will be used to automatically format and style the command header:

    ```
    <div class="cmdhead"></div>
    ```


3. Add a description of the command next. If you want to use the description in `Command.description`, add the following html:

    ```
    <div class="description"></div>
    ```

    However, be aware that if you want styling in the text such as making the text styled as code, then you need to add the `<code></code>` or other HTML tags to the description string in the Scala class.


4. If you want to add an auto-generated synopsis/usage statement from the args4j options, add the following html:

    ```
    <div class="synopsis"></div>
    ```


5. If you want to add an auto-generated command-line options description from the args4j options, add the following html:

    ```
    <div class="options"></div>
    ```

6. For any subsections, add a `<div class="cmdsubsection">` HTML tag at the beginning and format headers starting with `<h3>` in HTML or `###` in markdown. **Don't forget the `</div>` tag at the end of the markdown content!**

    ```
    <div class="cmdsubsection">
    ### My markdown header

    My Markdown text

    More Markdown text

    <div>
    ```

7. For examples, use the following html to label the example with a descriptive name. In addition, try to keep code blocks within 60-80 characters wide.

    ```
    <h4 class="example">Impute sex with a MAF threshold of 5%</h4>
    My example goes here
    ```


8. To view changes, use `gradle` to build the docs

    ```
    gradle --daemon createDocs
    ```

9. Change the directory to `build/docs/` and start a local web server using python
  
    ```
    cd build/docs/
    python -m SimpleHTTPServer 8000
    ```
    
10. The docs are viewable at `http://localhost:8000/`. To go directly to your command in the browser window, add a `#` and the command name (remember to replace spaces and slashes with underscores!)

11. If you make changes, be sure to rerun the gradle command to rebuild the docs before refreshing your browser.