# [MLIR](https://mlir.llvm.org) + [hail](https://hail.is) = 🚀🧬?

## Dependencies

You will need `cmake` and `ninja`. If you're on OS X, use https://brew.sh:

```sh
brew install cmake ninja
```

## Building/Installing LLVM and MLIR
Obviously, update the paths for your environment. `$HAIL_DIR` is the root of the
hail repository.

```sh
git clone https://github.com/llvm/llvm-project.git
mkdir llvm-project/build
cd llvm-project/build
git checkout llvmorg-15.0.3  # latest stable LLVM/MLIR release

# Some notes:
#     1. -G Ninja generates a build.ninja file rather than makefiles it's not
#        required but is recommended by LLVM
#     2. The CMAKE_INSTALL_PREFIX I put here is a subdirectory of the mlir-hail
#        (this repo's) root. If you do this, add that directory to
#        .git/info/exclude and it will be like adding it to a gitignore
#     3. On linux, using lld via -DLLVM_ENABLE_LLD=ON can speed up the build due
#        to faster linking.
#
# The -DLLVM_BUILD_EXAMPLES=ON flag is optional.
cmake ../llvm -G Ninja \
   -DLLVM_ENABLE_PROJECTS="clang;clang-tools-extra;mlir" \
   -DLLVM_BUILD_EXAMPLES=ON \
   -DLLVM_TARGETS_TO_BUILD="AArch64;X86;NVPTX;AMDGPU" \
   -DCMAKE_BUILD_TYPE=Release \
   -DLLVM_ENABLE_ASSERTIONS=ON \
   -DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++ \
   -DCMAKE_INSTALL_PREFIX=$HAIL_DIR/query/.dist/llvm \
   -DCMAKE_EXPORT_COMPILE_COMMANDS=1
ninja # this will take a while
ninja install
```


## Building Hail's native compiler

To set up the build, from this directory:

```sh
mkdir build
cd build
# You can set CC and CXX or use -DCMAKE_C_COMPILER and -DCMAKE_CXX_COMPILER to
# change the C and C++ compilers used to build this project.
cmake .. -G Ninja \
  -DCMAKE_BUILD_TYPE=RelWithDebInfo \
  -DMLIR_DIR=$HAIL_DIR/.dist/llvm/lib/cmake/mlir \
  -DCMAKE_EXPORT_COMPILE_COMMANDS=1 \
  -DLLVM_BUILD_BINARY_DIR=~/src/llvm-project/build/bin
  # ^ this argument is necessary to find llvm-lit and FileCheck for the tests
  #   they are skipped and a warning printed otherwise
```

To build:
```sh
cd build
ninja
```

To also run `clang-tidy` when building, pass `-DCMAKE_CXX_CLANG_TIDY=clang-tidy` or
`-DHAIL_USE_CLANG_TIDY=ON` to cmake. This can drastically increase build times,
so it may not be suitable for local development.

## Running the tests

To run all tests (from the `hail/query` directory):

```sh
ninja check-hail -C ./build
```

To run a single test (substitute `int.mlir` for the name of the test file to run):

```sh
cd ./build/test && llvm-lit ./int.mlir -v || true && cd -
```


## Setting up your editor

There are several [language servers](https://microsoft.github.io/language-server-protocol)
available that can vastly improve your editor experience when working with MLIR,
namely [`clangd` for C++ files](https://clangd.llvm.org) and [the MLIR-specific
language servers for `.mlir`, `.pdll`, and `.td` files](https://mlir.llvm.org/docs/Tools/MLIRLSP).
Below are instructions for setting them up with a few different editors.


### Visual Studio Code

If you're on macOS, you can install [VS Code](https://code.visualstudio.com/)
via Homebrew:

```sh
brew install --cask visual-studio-code
```

Open VS Code and install the
[clangd](https://marketplace.visualstudio.com/items?itemName=llvm-vs-code-extensions.vscode-clangd)
and [MLIR](https://marketplace.visualstudio.com/items?itemName=llvm-vs-code-extensions.vscode-mlir)
extensions, either by clicking the `Install` button on their webpages, or by
navigating to the `Extensions` icon in the bar on the left-hand side of the editor
window, searching for them by name, and clicking the `Install` button.

You will need [`jq`](https://stedolan.github.io/jq/) installed, if you don't
already have it. If you're on macOS, you can install it via Homebrew:

```sh
brew install jq
```

The location of the `settings.json` file on your platform can be found
[here](https://code.visualstudio.com/docs/getstarted/settings#_settings-file-locations).

```sh
# If you are not on macOS, replace this with the location of the `settings.json`
# file on your platform
VS_CODE_SETTINGS="$HOME/Library/Application Support/Code/User"
cp "${VS_CODE_SETTINGS}/settings.json" "${VS_CODE_SETTINGS}/settings.json.bak"
jq "
  .[\"mlir.pdll_server_path\"] = \"${LLVM_BUILD_BIN}/mlir-pdll-lsp-server\"
| .[\"mlir.server_path\"] = \"${LLVM_BUILD_BIN}/mlir-lsp-server\"
| .[\"mlir.tablegen_server_path\"] = \"${LLVM_BUILD_BIN}/tblgen-lsp-server\"
| .[\"mlir.tablegen_compilation_databases\"] = [
     \"${HAIL_NATIVE_COMPILER_BUILD}/tablegen_compile_commands.yml\"
   ]
" "${VS_CODE_SETTINGS}/settings.json.bak" > "${VS_CODE_SETTINGS}/settings.json"
```

Close and reopen VS Code, and the language servers should work; you can test this
by opening a `.td` file, right clicking a name `include`d from the MLIR project
(such as `Dialect`), and choosing `Go to Definition`.


### Neovim

If you're on macOS, you can install [Neovim](https://neovim.io/) via Homebrew:

```sh
brew install nvim
```

[Syntax highlighting](https://github.com/sheerun/vim-polyglot) and
[language server configuration](https://github.com/neovim/nvim-lspconfig) support
are provided by Neovim plugins. First, we'll install [packer](https://github.com/wbthomason/packer.nvim),
which will manage the installation of those plugins for us.

```sh
git clone --depth 1 https://github.com/wbthomason/packer.nvim ~/.local/share/nvim/site/pack/packer/start/packer.nvim
```

Next, place the following configuration in the `~/.config/nvim/init.lua` file. If you
already have a Neovim or Vim configuration written in Vimscript, you can migrate
it to Lua by copy-pasting it into the invocation of `vim.cmd`.

```lua
vim.cmd([[
" Vimscript configuration can be placed on multiple lines between these brackets
]])

local config = {
  globals = {
    mapleader = ' ',
  },
  options = {
    overrides = {
      number = true,
      signcolumn = 'number',
    },
  },
  keymap = {
    -- Add or replace keybinds here
    insert = {
      ['<C-Space>'] = '<C-x><C-o>',
    },
    normal = {
      ['<Leader>do'] = vim.diagnostic.open_float,
      ['[d'] = vim.diagnostic.goto_prev,
      [']d'] = vim.diagnostic.goto_next,
      ['<Leader>da'] = vim.lsp.buf.code_action,
      ['<Leader>dg'] = vim.lsp.buf.definition,
      ['<Leader>dk'] = vim.lsp.buf.hover,
      ['<Leader>ds'] = vim.lsp.buf.signature_help,
      ['<Leader>dr'] = vim.lsp.buf.rename,
      ['<Leader>df'] = vim.lsp.buf.references,
    },
  },
}

for k, v in pairs(config.globals) do vim.g[k] = v end
for k, v in pairs(config.options.overrides) do vim.opt[k] = v end
for k, v in pairs(config.keymap.insert) do vim.keymap.set('i', k, v, { noremap = true }) end
for k, v in pairs(config.keymap.normal) do vim.keymap.set('n', k, v, { noremap = true }) end

require('packer').startup(function(use)
  use 'neovim/nvim-lspconfig'
  use 'sheerun/vim-polyglot'
end)

local lspconfig = require('lspconfig')
lspconfig.clangd.setup({})
lspconfig.mlir_lsp_server.setup({})
lspconfig.mlir_pdll_lsp_server.setup({})
lspconfig.tblgen_lsp_server.setup({
  cmd = {
    "tblgen-lsp-server",
    "--tablegen-compilation-database=HAIL_NATIVE_COMPILER_BUILD/tablegen_compile_commands.yml",
  },
})
```

After saving the configuration file, run the following command to replace
`HAIL_NATIVE_COMPILER_BUILD` with its value:

```sh
sed -i .bak "s;HAIL_NATIVE_COMPILER_BUILD;${HAIL_NATIVE_COMPILER_BUILD};g" ~/.config/nvim/init.lua
```

This configuration provides keybinds for many of the major
[LSP features](https://neovim.io/doc/user/lsp.html#lsp-quickstart) that Neovim
supports. Keybinds can be added or replaced in the `config.keymap` table.

The absolute path to the `tablegen_compile_commands.yml` file will need to be
manually specified in order for the `tblgen-lsp-server` to function correctly.
Replace the `$HAIL_NATIVE_COMPILER_BUILD` variable in the configuration
with its value, which can be determined by running `echo $HAIL_NATIVE_COMPILER_BUILD`.

Close and reopen Neovim, and install the plugins:

```vim
:PackerInstall
```

The language servers should work; you can test this by opening a `.td` file,
placing the cursor on a name `include`d from the MLIR project (such as `Dialect`),
and calling the `vim.lsp.buf.definition` function (`<Space>gd` in the configuration provided).
