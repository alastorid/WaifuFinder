import SwiftUI
import AppKit

struct ImageLineageView:View {
    @ObservedObject var model:ImageLineageModel
    @ObservedObject var session:WaifuSession
    @AppStorage("rightPanelExpanded") private var showInspector=true
    private var findings:[String:LineageSafireSummary] { model.document.findings ?? [:] }
    private var ordered:[LineagePhoto] { LineageMaskRanking.ordered(model.document.photos,findings:findings) }
    private var lowestRatio:Double? { ordered.compactMap{findings[$0.id]?.ratio}.min() }
    private var completed:Int { ordered.filter{findings[$0.id]?.contentIdentity==$0.contentIdentity}.count }
    var body:some View {
        VStack(spacing:0) {
            HStack(spacing:12) {
                Label("SAFIRE",systemImage:"viewfinder").font(.headline)
                Text(ordered.count==1 ? "Photo" : "Selected group · \(ordered.count) photos").foregroundStyle(.secondary)
                Spacer()
                if model.running { ProgressView().controlSize(.small);Button("Stop",action:model.cancel) }
                else { Button("Refresh findings") { model.refresh(session:session) } }
            }.padding(12)
            Divider()
            HSplitView {
                ScrollView {
                    LazyVStack(alignment:.leading,spacing:18) {
                        Text("What SAFIRE found").font(.largeTitle.weight(.semibold))
                        Text("Lowest forged-mask area first. Red fill and bright outlines show the flagged part; the separate white mask shows its shape.")
                            .font(.body).foregroundStyle(.secondary)
                        Text(model.progress).font(.caption).foregroundStyle(.secondary)
                        if let issue=model.issue { Text(issue).font(.callout).foregroundStyle(.orange).textSelection(.enabled) }
                        ForEach(Array(ordered.enumerated()),id:\.element.id) { index,photo in
                            SafireImageCard(photo:photo,summary:findings[photo.id],first:ordered.count>1 && findings[photo.id]?.ratio==lowestRatio,complete:completed==ordered.count,
                                selected:model.document.selectedPhotoID==photo.id) { model.select(photo.id) }
                            if model.document.selectedPhotoID==photo.id,let material=model.overlay,!material.regions.isEmpty {
                                DisclosureGroup("Why these regions are highlighted") {
                                    VStack(alignment:.leading,spacing:12) {
                                        Text("SAFIRE’s learned spatial features produce this manipulation map. Regions above the display threshold are highlighted. Natural content can also trigger the model; it does not name the editing application or predict chronology.")
                                            .font(.callout).foregroundStyle(.secondary)
                                        ForEach(material.regions) { region in
                                            LineageFindingCard(region:region,evidence:material.result,image:model.storyImage,
                                                selected:model.selectedRegionID==region.id) { model.selectRegion(region.id) }
                                        }
                                    }.padding(.top,8)
                                }
                            }
                        }
                        if ordered.isEmpty { Text("Reading the selected photos…").foregroundStyle(.secondary) }
                    }.padding(20).frame(maxWidth:1100,alignment:.leading).frame(maxWidth:.infinity)
                }.frame(minWidth:460)
                if showInspector {
                    ScrollView {
                        VStack(alignment:.leading,spacing:12) {
                            Text("Photo information").font(.headline)
                            if let photo=model.selected { Text(photo.name).font(.headline);Button("Show in Photos") { PhotosAppIntegration.reveal(photo.url) } }
                            PhotoMetadataInspectorView(metadata:model.metadata,isLoading:model.metadataLoading,errorMessage:nil)
                            Button(role:.destructive) { model.requestDelete() } label: { Label("Delete Photo from Library…",systemImage:"trash") }
                                .disabled(!model.canDelete)
                        }.padding(14)
                    }.frame(minWidth:250,idealWidth:285,maxWidth:330)
                }
            }
        }.background(Color(nsColor:.windowBackgroundColor))
        .confirmationDialog("Delete from your Photos library?",isPresented:$model.showDeleteConfirmation,titleVisibility:.visible) {
            Button("Delete Photo from Library",role:.destructive) { model.deleteConfirmed() }
            Button("Cancel",role:.cancel) { model.deletionTargetsReset() }
        } message: {
            Text(model.deletionTargets.map(\.name).joined(separator:"\n")+"\n\nThis deletes the photo from Photos and every album, and removes its SAFIRE evidence. Photos may ask for authorization and manages recovery in Recently Deleted.")
        }
    }
}

private struct LineageFindingCard: View {
    let region:LineageRegion
    let evidence:SafireSpatialResult
    let image:CGImage?
    let selected:Bool
    let select:()->Void
    private var location:String {
        let x=region.bounds.midX/CGFloat(evidence.width),y=region.bounds.midY/CGFloat(evidence.height)
        return (y<1.0/3 ? "upper" : y>2.0/3 ? "lower" : "middle")+" "+(x<1.0/3 ? "left" : x>2.0/3 ? "right" : "center")
    }
    private var crop:CGImage? {
        guard let image else { return nil }
        return image.cropping(to:evidence.nativeBounds(region,size:CGSize(width:image.width,height:image.height)).integral.intersection(CGRect(x:0,y:0,width:image.width,height:image.height)))
    }
    var body: some View {
        Button(action:select) {
            HStack(alignment:.top,spacing:14) {
                if let crop { Image(decorative:crop,scale:1).resizable().scaledToFit().frame(width:130,height:110).overlay(Rectangle().stroke(.orange,lineWidth:1)) }
                VStack(alignment:.leading,spacing:8) {
                    Text("Region in the \(location)").font(.headline)
                    Text("SAFIRE’s manipulation map highlights a connected area here. The crop shows its bounds, and the map on the photograph shows the precise spatial evidence.")
                    Text("Why it is highlighted: neighboring mask pixels exceed the 0.5 display threshold. The model does not name the object, the editing application, or when a change occurred.")
                        .font(.callout).foregroundStyle(.secondary)
                }.frame(maxWidth:.infinity,alignment:.leading)
            }.padding(14).background(selected ? Color.accentColor.opacity(0.1) : Color.secondary.opacity(0.05),in:RoundedRectangle(cornerRadius:10))
        }.buttonStyle(.plain)
    }
}

