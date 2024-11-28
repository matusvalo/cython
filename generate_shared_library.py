from Cython.Compiler import Options
from Cython.Compiler import Pipeline
from Cython.Compiler import Errors
from Cython.Compiler import Main
from Cython.Compiler import Symtab
from Cython.Compiler import MemoryView
from Cython.Compiler.StringEncoding import EncodedString
from Cython.Compiler.Scanning import StringSourceDescriptor, FileSourceDescriptor

Errors.init_thread()
Options.generate_cyshared = True

options = Options.CompilationOptions()
context = Main.Context.from_options(options)
scope = Symtab.ModuleScope('cyshared', parent_module = None, context = context, is_package=False)

source = StringSourceDescriptor("cyshared", '')
source.filename = 'cyshared.pyx'
# source = FileSourceDescriptor('cyshared.pxd')
comp_src = Main.CompilationSource(source, EncodedString('cyshared'), '.')
result = Main.create_default_resultobj(comp_src, options)

pipeline = Pipeline.create_dummy_pipeline(context, scope, options, result)
result = Pipeline.run_pipeline(pipeline, comp_src)
print(result)

# x86_64-linux-gnu-gcc -Wsign-compare -DNDEBUG -g -fwrapv -O2 -Wall -g -fstack-protector-strong -Wformat -Werror=format-security -g -fwrapv -O2 -g -fstack-protector-strong -Wformat -Werror=format-security -Wdate-time -D_FORTIFY_SOURCE=2 -fPIC -I/root/dev/cython311/include -I/usr/include/python3.11 -c /root/dev/cython/test/cyshared.c -o /root/dev/cython/test/cyshared.o
# x86_64-linux-gnu-gcc -shared -Wl,-O1 -Wl,-Bsymbolic-functions -g -fwrapv -O2 -g -fwrapv -O2 -g -fstack-protector-strong -Wformat -Werror=format-security -Wdate-time -D_FORTIFY_SOURCE=2 /root/dev/cython/test/cyshared.o -o /root/dev/cython/test/cyshared.cpython-311-x86_64-linux-gnu.so
