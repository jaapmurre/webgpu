
var Module = (function() {
  var _scriptDir = typeof document !== 'undefined' && document.currentScript ? document.currentScript.src : undefined;
  return (
function(Module) {
  Module = Module || {};

var e;e||(e=typeof Module !== 'undefined' ? Module : {});e.compileGLSLZeroCopy=function(a,b,c){c=!!c;if("vertex"===b)var d=0;else if("fragment"===b)d=4;else if("compute"===b)d=5;else throw Error("shader_stage must be 'vertex', 'fragment', or 'compute'");b=e._malloc(4);var g=e._malloc(4),f=aa([a,d,c,b,g]);c=ba(b);a=ba(g);e._free(b);e._free(g);if(0===f)throw Error("GLSL compilation failed");b={};g=c/4;b.data=e.HEAPU32.subarray(g,g+a);b.free=function(){e._destroy_output_buffer(f)};return b};
e.compileGLSL=function(a,b,c){a=e.compileGLSLZeroCopy(a,b,c);b=a.data.slice();a.free();return b};var l={},p;for(p in e)e.hasOwnProperty(p)&&(l[p]=e[p]);var r="./this.program",t="",ca,u,w,x;t=__dirname+"/";ca=function(a){w||(w=require("fs"));x||(x=require("path"));a=x.normalize(a);return w.readFileSync(a,null)};u=function(a){a=ca(a);a.buffer||(a=new Uint8Array(a));a.buffer||y("Assertion failed: undefined");return a};1<process.argv.length&&(r=process.argv[1].replace(/\\/g,"/"));process.argv.slice(2);
process.on("uncaughtException",function(a){throw a;});process.on("unhandledRejection",y);e.inspect=function(){return"[Emscripten Module object]"};var da=e.print||console.log.bind(console),z=e.printErr||console.warn.bind(console);for(p in l)l.hasOwnProperty(p)&&(e[p]=l[p]);l=null;e.thisProgram&&(r=e.thisProgram);var A;e.wasmBinary&&(A=e.wasmBinary);"object"!==typeof WebAssembly&&z("no native wasm support detected");
function ba(a){var b="i32";"*"===b.charAt(b.length-1)&&(b="i32");switch(b){case "i1":return B[a>>0];case "i8":return B[a>>0];case "i16":return C[a>>1];case "i32":return D[a>>2];case "i64":return D[a>>2];case "float":return E[a>>2];case "double":return F[a>>3];default:y("invalid type for getValue: "+b)}return null}var G,ea=new WebAssembly.Table({initial:861,maximum:861,element:"anyfunc"}),fa=!1;
function ha(){var a=e._convert_glsl_to_spirv;a||y("Assertion failed: Cannot call unknown function convert_glsl_to_spirv, make sure it is exported");return a}
function aa(a){var b=["string","number","boolean","number","number"],c={string:function(a){var b=0;if(null!==a&&void 0!==a&&0!==a){var c=(a.length<<2)+1;b=ia(c);H(a,I,b,c)}return b},array:function(a){var b=ia(a.length);B.set(a,b);return b}},d=ha(),g=[],f=0;if(a)for(var h=0;h<a.length;h++){var k=c[b[h]];k?(0===f&&(f=ja()),g[h]=k(a[h])):g[h]=a[h]}a=d.apply(null,g);0!==f&&ka(f);return a}var la="undefined"!==typeof TextDecoder?new TextDecoder("utf8"):void 0;
function ma(a,b,c){var d=b+c;for(c=b;a[c]&&!(c>=d);)++c;if(16<c-b&&a.subarray&&la)return la.decode(a.subarray(b,c));for(d="";b<c;){var g=a[b++];if(g&128){var f=a[b++]&63;if(192==(g&224))d+=String.fromCharCode((g&31)<<6|f);else{var h=a[b++]&63;g=224==(g&240)?(g&15)<<12|f<<6|h:(g&7)<<18|f<<12|h<<6|a[b++]&63;65536>g?d+=String.fromCharCode(g):(g-=65536,d+=String.fromCharCode(55296|g>>10,56320|g&1023))}}else d+=String.fromCharCode(g)}return d}function na(a){return a?ma(I,a,void 0):""}
function H(a,b,c,d){if(0<d){d=c+d-1;for(var g=0;g<a.length;++g){var f=a.charCodeAt(g);if(55296<=f&&57343>=f){var h=a.charCodeAt(++g);f=65536+((f&1023)<<10)|h&1023}if(127>=f){if(c>=d)break;b[c++]=f}else{if(2047>=f){if(c+1>=d)break;b[c++]=192|f>>6}else{if(65535>=f){if(c+2>=d)break;b[c++]=224|f>>12}else{if(c+3>=d)break;b[c++]=240|f>>18;b[c++]=128|f>>12&63}b[c++]=128|f>>6&63}b[c++]=128|f&63}}b[c]=0}}
function oa(a){for(var b=0,c=0;c<a.length;++c){var d=a.charCodeAt(c);55296<=d&&57343>=d&&(d=65536+((d&1023)<<10)|a.charCodeAt(++c)&1023);127>=d?++b:b=2047>=d?b+2:65535>=d?b+3:b+4}return b}"undefined"!==typeof TextDecoder&&new TextDecoder("utf-16le");function pa(a){0<a%65536&&(a+=65536-a%65536);return a}var buffer,B,I,C,qa,D,J,E,F;
function ra(a){buffer=a;e.HEAP8=B=new Int8Array(a);e.HEAP16=C=new Int16Array(a);e.HEAP32=D=new Int32Array(a);e.HEAPU8=I=new Uint8Array(a);e.HEAPU16=qa=new Uint16Array(a);e.HEAPU32=J=new Uint32Array(a);e.HEAPF32=E=new Float32Array(a);e.HEAPF64=F=new Float64Array(a)}var sa=e.TOTAL_MEMORY||16777216;e.wasmMemory?G=e.wasmMemory:G=new WebAssembly.Memory({initial:sa/65536});G&&(buffer=G.buffer);sa=buffer.byteLength;ra(buffer);D[79464]=5560896;
function K(a){for(;0<a.length;){var b=a.shift();if("function"==typeof b)b();else{var c=b.T;"number"===typeof c?void 0===b.R?e.dynCall_v(c):e.dynCall_vi(c,b.R):c(void 0===b.R?null:b.R)}}}var ta=[],ua=[],va=[],wa=[];function xa(){var a=e.preRun.shift();ta.unshift(a)}var L=0,ya=null,M=null;e.preloadedImages={};e.preloadedAudios={};function y(a){if(e.onAbort)e.onAbort(a);da(a);z(a);fa=!0;throw new WebAssembly.RuntimeError("abort("+a+"). Build with -s ASSERTIONS=1 for more info.");}var N="glslang.wasm";
if(String.prototype.startsWith?!N.startsWith("data:application/octet-stream;base64,"):0!==N.indexOf("data:application/octet-stream;base64,")){var za=N;N=e.locateFile?e.locateFile(za,t):t+za}ua.push({T:function(){Aa()}});var Ba=[null,[],[]],Ca=0;function Da(){Ca+=4;return D[Ca-4>>2]}var Ea={};function Fa(a){switch(a){case 1:return 0;case 2:return 1;case 4:return 2;case 8:return 3;default:throw new TypeError("Unknown type size: "+a);}}var Ga=void 0;
function O(a){for(var b="";I[a];)b+=Ga[I[a++]];return b}var Ha={},Ia={},Ja={};function La(a,b){if(void 0===a)a="_unknown";else{a=a.replace(/[^a-zA-Z0-9_]/g,"$");var c=a.charCodeAt(0);a=48<=c&&57>=c?"_"+a:a}return(new Function("body","return function "+a+'() {\n    "use strict";    return body.apply(this, arguments);\n};\n'))(b)}
function Ma(a){var b=Error,c=La(a,function(b){this.name=a;this.message=b;b=Error(b).stack;void 0!==b&&(this.stack=this.toString()+"\n"+b.replace(/^Error(:[^\n]*)?\n/,""))});c.prototype=Object.create(b.prototype);c.prototype.constructor=c;c.prototype.toString=function(){return void 0===this.message?this.name:this.name+": "+this.message};return c}var Na=void 0;function P(a){throw new Na(a);}
function Q(a,b,c){c=c||{};if(!("argPackAdvance"in b))throw new TypeError("registerType registeredInstance requires argPackAdvance");var d=b.name;a||P('type "'+d+'" must have a positive integer typeid pointer');if(Ia.hasOwnProperty(a)){if(c.U)return;P("Cannot register type '"+d+"' twice")}Ia[a]=b;delete Ja[a];Ha.hasOwnProperty(a)&&(b=Ha[a],delete Ha[a],b.forEach(function(a){a()}))}var Oa=[],R=[{},{value:void 0},{value:null},{value:!0},{value:!1}];
function Pa(a){switch(a){case void 0:return 1;case null:return 2;case !0:return 3;case !1:return 4;default:var b=Oa.length?Oa.pop():R.length;R[b]={W:1,value:a};return b}}function Qa(a){return this.fromWireType(J[a>>2])}function Ra(a){if(null===a)return"null";var b=typeof a;return"object"===b||"array"===b||"function"===b?a.toString():""+a}
function Sa(a,b){switch(b){case 2:return function(a){return this.fromWireType(E[a>>2])};case 3:return function(a){return this.fromWireType(F[a>>3])};default:throw new TypeError("Unknown float type: "+a);}}
function Ta(a,b,c){switch(b){case 0:return c?function(a){return B[a]}:function(a){return I[a]};case 1:return c?function(a){return C[a>>1]}:function(a){return qa[a>>1]};case 2:return c?function(a){return D[a>>2]}:function(a){return J[a>>2]};default:throw new TypeError("Unknown integer type: "+a);}}var Ua={};
function Va(){if(!Wa){var a={USER:"web_user",LOGNAME:"web_user",PATH:"/",PWD:"/",HOME:"/home/web_user",LANG:("object"===typeof navigator&&navigator.languages&&navigator.languages[0]||"C").replace("-","_")+".UTF-8",_:r},b;for(b in Ua)a[b]=Ua[b];var c=[];for(b in a)c.push(b+"="+a[b]);Wa=c}return Wa}var Wa;function S(a){return 0===a%4&&(0!==a%100||0===a%400)}function Xa(a,b){for(var c=0,d=0;d<=b;c+=a[d++]);return c}var T=[31,29,31,30,31,30,31,31,30,31,30,31],U=[31,28,31,30,31,30,31,31,30,31,30,31];
function V(a,b){for(a=new Date(a.getTime());0<b;){var c=a.getMonth(),d=(S(a.getFullYear())?T:U)[c];if(b>d-a.getDate())b-=d-a.getDate()+1,a.setDate(1),11>c?a.setMonth(c+1):(a.setMonth(0),a.setFullYear(a.getFullYear()+1));else{a.setDate(a.getDate()+b);break}}return a}
function Ya(a,b,c,d){function g(a,b,c){for(a="number"===typeof a?a.toString():a||"";a.length<b;)a=c[0]+a;return a}function f(a,b){return g(a,b,"0")}function h(a,b){function c(a){return 0>a?-1:0<a?1:0}var d;0===(d=c(a.getFullYear()-b.getFullYear()))&&0===(d=c(a.getMonth()-b.getMonth()))&&(d=c(a.getDate()-b.getDate()));return d}function k(a){switch(a.getDay()){case 0:return new Date(a.getFullYear()-1,11,29);case 1:return a;case 2:return new Date(a.getFullYear(),0,3);case 3:return new Date(a.getFullYear(),
0,2);case 4:return new Date(a.getFullYear(),0,1);case 5:return new Date(a.getFullYear()-1,11,31);case 6:return new Date(a.getFullYear()-1,11,30)}}function q(a){a=V(new Date(a.J+1900,0,1),a.P);var b=k(new Date(a.getFullYear()+1,0,4));return 0>=h(k(new Date(a.getFullYear(),0,4)),a)?0>=h(b,a)?a.getFullYear()+1:a.getFullYear():a.getFullYear()-1}var m=D[d+40>>2];d={Z:D[d>>2],Y:D[d+4>>2],N:D[d+8>>2],M:D[d+12>>2],K:D[d+16>>2],J:D[d+20>>2],O:D[d+24>>2],P:D[d+28>>2],ia:D[d+32>>2],X:D[d+36>>2],$:m?na(m):""};
c=na(c);m={"%c":"%a %b %d %H:%M:%S %Y","%D":"%m/%d/%y","%F":"%Y-%m-%d","%h":"%b","%r":"%I:%M:%S %p","%R":"%H:%M","%T":"%H:%M:%S","%x":"%m/%d/%y","%X":"%H:%M:%S","%Ec":"%c","%EC":"%C","%Ex":"%m/%d/%y","%EX":"%H:%M:%S","%Ey":"%y","%EY":"%Y","%Od":"%d","%Oe":"%e","%OH":"%H","%OI":"%I","%Om":"%m","%OM":"%M","%OS":"%S","%Ou":"%u","%OU":"%U","%OV":"%V","%Ow":"%w","%OW":"%W","%Oy":"%y"};for(var n in m)c=c.replace(new RegExp(n,"g"),m[n]);var v="Sunday Monday Tuesday Wednesday Thursday Friday Saturday".split(" "),
Ka="January February March April May June July August September October November December".split(" ");m={"%a":function(a){return v[a.O].substring(0,3)},"%A":function(a){return v[a.O]},"%b":function(a){return Ka[a.K].substring(0,3)},"%B":function(a){return Ka[a.K]},"%C":function(a){return f((a.J+1900)/100|0,2)},"%d":function(a){return f(a.M,2)},"%e":function(a){return g(a.M,2," ")},"%g":function(a){return q(a).toString().substring(2)},"%G":function(a){return q(a)},"%H":function(a){return f(a.N,2)},
"%I":function(a){a=a.N;0==a?a=12:12<a&&(a-=12);return f(a,2)},"%j":function(a){return f(a.M+Xa(S(a.J+1900)?T:U,a.K-1),3)},"%m":function(a){return f(a.K+1,2)},"%M":function(a){return f(a.Y,2)},"%n":function(){return"\n"},"%p":function(a){return 0<=a.N&&12>a.N?"AM":"PM"},"%S":function(a){return f(a.Z,2)},"%t":function(){return"\t"},"%u":function(a){return a.O||7},"%U":function(a){var b=new Date(a.J+1900,0,1),c=0===b.getDay()?b:V(b,7-b.getDay());a=new Date(a.J+1900,a.K,a.M);return 0>h(c,a)?f(Math.ceil((31-
c.getDate()+(Xa(S(a.getFullYear())?T:U,a.getMonth()-1)-31)+a.getDate())/7),2):0===h(c,b)?"01":"00"},"%V":function(a){var b=k(new Date(a.J+1900,0,4)),c=k(new Date(a.J+1901,0,4)),d=V(new Date(a.J+1900,0,1),a.P);return 0>h(d,b)?"53":0>=h(c,d)?"01":f(Math.ceil((b.getFullYear()<a.J+1900?a.P+32-b.getDate():a.P+1-b.getDate())/7),2)},"%w":function(a){return a.O},"%W":function(a){var b=new Date(a.J,0,1),c=1===b.getDay()?b:V(b,0===b.getDay()?1:7-b.getDay()+1);a=new Date(a.J+1900,a.K,a.M);return 0>h(c,a)?f(Math.ceil((31-
c.getDate()+(Xa(S(a.getFullYear())?T:U,a.getMonth()-1)-31)+a.getDate())/7),2):0===h(c,b)?"01":"00"},"%y":function(a){return(a.J+1900).toString().substring(2)},"%Y":function(a){return a.J+1900},"%z":function(a){a=a.X;var b=0<=a;a=Math.abs(a)/60;return(b?"+":"-")+String("0000"+(a/60*100+a%60)).slice(-4)},"%Z":function(a){return a.$},"%%":function(){return"%"}};for(n in m)0<=c.indexOf(n)&&(c=c.replace(new RegExp(n,"g"),m[n](d)));n=Za(c);if(n.length>b)return 0;B.set(n,a);return n.length-1}
for(var $a=Array(256),W=0;256>W;++W)$a[W]=String.fromCharCode(W);Ga=$a;Na=e.BindingError=Ma("BindingError");e.InternalError=Ma("InternalError");e.count_emval_handles=function(){for(var a=0,b=5;b<R.length;++b)void 0!==R[b]&&++a;return a};e.get_first_emval=function(){for(var a=5;a<R.length;++a)if(void 0!==R[a])return R[a];return null};function Za(a){var b=Array(oa(a)+1);H(a,b,0,b.length);return b}
var bb={j:function(){},g:function(){e.___errno_location&&(D[e.___errno_location()>>2]=63);return-1},v:function(a,b){Ca=b;try{var c=Da();var d=Da();if(-1===c||0===d)var g=-28;else{var f=Ea.V[c];if(f&&d===f.fa){var h=(void 0).da(f.fd);Ea.ba(c,h,d,f.flags);(void 0).ha(h);Ea.V[c]=null;f.aa&&X(f.ga)}g=0}return g}catch(k){return y(k),-k.S}},d:function(){},s:function(a,b,c,d,g){var f=Fa(c);b=O(b);Q(a,{name:b,fromWireType:function(a){return!!a},toWireType:function(a,b){return b?d:g},argPackAdvance:8,readValueFromPointer:function(a){if(1===
c)var d=B;else if(2===c)d=C;else if(4===c)d=D;else throw new TypeError("Unknown boolean type size: "+b);return this.fromWireType(d[a>>f])},L:null})},q:function(a,b){b=O(b);Q(a,{name:b,fromWireType:function(a){var b=R[a].value;4<a&&0===--R[a].W&&(R[a]=void 0,Oa.push(a));return b},toWireType:function(a,b){return Pa(b)},argPackAdvance:8,readValueFromPointer:Qa,L:null})},e:function(a,b,c){c=Fa(c);b=O(b);Q(a,{name:b,fromWireType:function(a){return a},toWireType:function(a,b){if("number"!==typeof b&&"boolean"!==
typeof b)throw new TypeError('Cannot convert "'+Ra(b)+'" to '+this.name);return b},argPackAdvance:8,readValueFromPointer:Sa(b,c),L:null})},b:function(a,b,c,d,g){function f(a){return a}b=O(b);-1===g&&(g=4294967295);var h=Fa(c);if(0===d){var k=32-8*c;f=function(a){return a<<k>>>k}}var q=-1!=b.indexOf("unsigned");Q(a,{name:b,fromWireType:f,toWireType:function(a,c){if("number"!==typeof c&&"boolean"!==typeof c)throw new TypeError('Cannot convert "'+Ra(c)+'" to '+this.name);if(c<d||c>g)throw new TypeError('Passing a number "'+
Ra(c)+'" from JS side to C/C++ side to an argument of type "'+b+'", which is outside the valid range ['+d+", "+g+"]!");return q?c>>>0:c|0},argPackAdvance:8,readValueFromPointer:Ta(b,h,0!==d),L:null})},a:function(a,b,c){function d(a){a>>=2;var b=J;return new g(b.buffer,b[a+1],b[a])}var g=[Int8Array,Uint8Array,Int16Array,Uint16Array,Int32Array,Uint32Array,Float32Array,Float64Array][b];c=O(c);Q(a,{name:c,fromWireType:d,argPackAdvance:8,readValueFromPointer:d},{U:!0})},f:function(a,b){b=O(b);var c="std::string"===
b;Q(a,{name:b,fromWireType:function(a){var b=J[a>>2];if(c){var d=I[a+4+b],h=0;0!=d&&(h=d,I[a+4+b]=0);var k=a+4;for(d=0;d<=b;++d){var q=a+4+d;if(0==I[q]){k=na(k);if(void 0===m)var m=k;else m+=String.fromCharCode(0),m+=k;k=q+1}}0!=h&&(I[a+4+b]=h)}else{m=Array(b);for(d=0;d<b;++d)m[d]=String.fromCharCode(I[a+4+d]);m=m.join("")}X(a);return m},toWireType:function(a,b){b instanceof ArrayBuffer&&(b=new Uint8Array(b));var d="string"===typeof b;d||b instanceof Uint8Array||b instanceof Uint8ClampedArray||b instanceof
Int8Array||P("Cannot pass non-string to std::string");var g=(c&&d?function(){return oa(b)}:function(){return b.length})(),k=ab(4+g+1);J[k>>2]=g;if(c&&d)H(b,I,k+4,g+1);else if(d)for(d=0;d<g;++d){var q=b.charCodeAt(d);255<q&&(X(k),P("String has UTF-16 code units that do not fit in 8 bits"));I[k+4+d]=q}else for(d=0;d<g;++d)I[k+4+d]=b[d];null!==a&&a.push(X,k);return k},argPackAdvance:8,readValueFromPointer:Qa,L:function(a){X(a)}})},r:function(a,b,c){c=O(c);if(2===b){var d=function(){return qa};var g=
1}else 4===b&&(d=function(){return J},g=2);Q(a,{name:c,fromWireType:function(a){for(var b=d(),c=J[a>>2],f=Array(c),m=a+4>>g,n=0;n<c;++n)f[n]=String.fromCharCode(b[m+n]);X(a);return f.join("")},toWireType:function(a,c){var f=c.length,h=ab(4+f*b),m=d();J[h>>2]=f;for(var n=h+4>>g,v=0;v<f;++v)m[n+v]=c.charCodeAt(v);null!==a&&a.push(X,h);return h},argPackAdvance:8,readValueFromPointer:Qa,L:function(a){X(a)}})},t:function(a,b){b=O(b);Q(a,{ea:!0,name:b,argPackAdvance:0,fromWireType:function(){},toWireType:function(){}})},
c:function(){y()},n:function(a,b,c){I.set(I.subarray(b,b+c),a)},o:function(a){if(2147418112<a)return!1;for(var b=Math.max(B.length,16777216);b<a;)536870912>=b?b=pa(2*b):b=Math.min(pa((3*b+2147483648)/4),2147418112);a:{try{G.grow(b-buffer.byteLength+65535>>16);ra(G.buffer);var c=1;break a}catch(d){}c=void 0}return c?!0:!1},h:function(a,b){var c=0;Va().forEach(function(d,g){var f=b+c;g=D[a+4*g>>2]=f;for(f=0;f<d.length;++f)B[g++>>0]=d.charCodeAt(f);B[g>>0]=0;c+=d.length+1});return 0},i:function(a,b){var c=
Va();D[a>>2]=c.length;var d=0;c.forEach(function(a){d+=a.length+1});D[b>>2]=d;return 0},l:function(){return 0},m:function(){return 0},k:function(a,b,c,d){try{for(var g=0,f=0;f<c;f++){for(var h=D[b+8*f>>2],k=D[b+(8*f+4)>>2],q=0;q<k;q++){var m=I[h+q],n=Ba[a];0===m||10===m?((1===a?da:z)(ma(n,0)),n.length=0):n.push(m)}g+=k}D[d>>2]=g;return 0}catch(v){return y(v),v.S}},memory:G,w:function(){},p:function(){},u:function(a,b,c,d){return Ya(a,b,c,d)},table:ea},Y=function(){function a(a){e.asm=a.exports;L--;
e.monitorRunDependencies&&e.monitorRunDependencies(L);0==L&&(null!==ya&&(clearInterval(ya),ya=null),M&&(a=M,M=null,a()))}var b={env:bb,wasi_unstable:bb};L++;e.monitorRunDependencies&&e.monitorRunDependencies(L);if(e.instantiateWasm)try{return e.instantiateWasm(b,a)}catch(c){return z("Module.instantiateWasm callback failed with error: "+c),!1}(function(){try{a:{try{if(A){var c=new Uint8Array(A);break a}if(u){c=u(N);break a}throw"sync fetching of the wasm failed: you can preload it to Module['wasmBinary'] manually, or emcc.py will do that for you when generating HTML (but not JS)";
}catch(f){y(f)}c=void 0}var d=new WebAssembly.Module(c);var g=new WebAssembly.Instance(d,b)}catch(f){throw g=f.toString(),z("failed to compile wasm module: "+g),(0<=g.indexOf("imported Memory")||0<=g.indexOf("memory import"))&&z("Memory size incompatibility issues may be due to changing TOTAL_MEMORY at runtime to something too large. Use ALLOW_MEMORY_GROWTH to allow any size memory (and also make sure not to set TOTAL_MEMORY at runtime to something smaller than it was at compile time)."),f;}a(g,d)})();
return e.asm}(),Aa=e.___wasm_call_ctors=Y.x;e._convert_glsl_to_spirv=Y.y;e._destroy_output_buffer=Y.z;var ab=e._malloc=Y.A,X=e._free=Y.B;e.___getTypeName=Y.C;e.___embind_register_native_and_builtin_types=Y.D;var ja=e.stackSave=Y.E,ia=e.stackAlloc=Y.F,ka=e.stackRestore=Y.G;e.dynCall_vi=Y.H;e.dynCall_v=Y.I;e.asm=Y;var Z;e.then=function(a){if(Z)a(e);else{var b=e.onRuntimeInitialized;e.onRuntimeInitialized=function(){b&&b();a(e)}}return e};M=function cb(){Z||db();Z||(M=cb)};
function db(){function a(){if(!Z&&(Z=!0,!fa)){K(ua);K(va);if(e.onRuntimeInitialized)e.onRuntimeInitialized();if(e.postRun)for("function"==typeof e.postRun&&(e.postRun=[e.postRun]);e.postRun.length;){var a=e.postRun.shift();wa.unshift(a)}K(wa)}}if(!(0<L)){if(e.preRun)for("function"==typeof e.preRun&&(e.preRun=[e.preRun]);e.preRun.length;)xa();K(ta);0<L||(e.setStatus?(e.setStatus("Running..."),setTimeout(function(){setTimeout(function(){e.setStatus("")},1);a()},1)):a())}}e.run=db;
if(e.preInit)for("function"==typeof e.preInit&&(e.preInit=[e.preInit]);0<e.preInit.length;)e.preInit.pop()();db();


  return Module
}
);
})();
if (typeof exports === 'object' && typeof module === 'object')
      module.exports = Module;
    else if (typeof define === 'function' && define['amd'])
      define([], function() { return Module; });
    else if (typeof exports === 'object')
      exports["Module"] = Module;
    