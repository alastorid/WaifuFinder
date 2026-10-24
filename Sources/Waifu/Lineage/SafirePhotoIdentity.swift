import Foundation
import Photos

enum SafirePhotoIdentity {
    static func make(_ item:PhotosMedia.Item)->LineagePhoto {
        let resource=PHAssetResource.assetResources(for:item.asset).first{$0.type == .fullSizePhoto}
            ?? PHAssetResource.assetResources(for:item.asset).first{$0.type == .photo}
        let name=resource?.originalFilename ?? "Photo"
        let revision="\(item.asset.localIdentifier)|\(item.asset.modificationDate?.timeIntervalSince1970 ?? 0)|\(item.asset.pixelWidth)x\(item.asset.pixelHeight)"
        return .init(id:item.asset.localIdentifier,contentIdentity:LineageIdentity.hash(revision),name:name,
            width:item.asset.pixelWidth,height:item.asset.pixelHeight,format:(name as NSString).pathExtension.uppercased(),
            url:item.url,offset:item.offset,photosDate:item.asset.creationDate)
    }
}
