import fs from 'node:fs';import cp from 'node:child_process';import path from 'node:path';
const root=path.resolve('.build/wd-native/staged-source');
function files(p){return fs.readdirSync(p,{withFileTypes:true}).flatMap(e=>e.isDirectory()?files(path.join(p,e.name)):e.name.endsWith('.swift')?[path.join(p,e.name)]:[]);}
const sdk=cp.execFileSync('xcrun',['--sdk','macosx','--show-sdk-path'],{encoding:'utf8'}).trim();
const frameworks=['AppKit','AVFoundation','AVKit','CoreMedia','Vision','Photos','CoreImage','CoreML','UniformTypeIdentifiers'];
const args=['swiftc','-target','arm64-apple-macosx15.0','-sdk',sdk,'-parse-as-library','-swift-version','6','-warnings-as-errors','-O','-g','-whole-module-optimization','-o',path.resolve('.build/wd-native/checkpoint/WaifuFinder'),...files(path.join(root,'Sources')).sort(),...frameworks.flatMap(f=>['-framework',f])];
const log=fs.openSync('.build/wd-native/staged-app-build.log','w');const r=cp.spawnSync('xcrun',args,{stdio:['ignore',log,log]});fs.closeSync(log);process.exit(r.status??1);
