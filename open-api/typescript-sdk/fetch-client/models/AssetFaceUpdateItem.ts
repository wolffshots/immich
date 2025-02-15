/* tslint:disable */
/* eslint-disable */
/**
 * Immich
 * Immich API
 *
 * The version of the OpenAPI document: 1.94.1
 * 
 *
 * NOTE: This class is auto generated by OpenAPI Generator (https://openapi-generator.tech).
 * https://openapi-generator.tech
 * Do not edit the class manually.
 */

import { exists, mapValues } from '../runtime';
/**
 * 
 * @export
 * @interface AssetFaceUpdateItem
 */
export interface AssetFaceUpdateItem {
    /**
     * 
     * @type {string}
     * @memberof AssetFaceUpdateItem
     */
    assetId: string;
    /**
     * 
     * @type {string}
     * @memberof AssetFaceUpdateItem
     */
    personId: string;
}

/**
 * Check if a given object implements the AssetFaceUpdateItem interface.
 */
export function instanceOfAssetFaceUpdateItem(value: object): boolean {
    let isInstance = true;
    isInstance = isInstance && "assetId" in value;
    isInstance = isInstance && "personId" in value;

    return isInstance;
}

export function AssetFaceUpdateItemFromJSON(json: any): AssetFaceUpdateItem {
    return AssetFaceUpdateItemFromJSONTyped(json, false);
}

export function AssetFaceUpdateItemFromJSONTyped(json: any, ignoreDiscriminator: boolean): AssetFaceUpdateItem {
    if ((json === undefined) || (json === null)) {
        return json;
    }
    return {
        
        'assetId': json['assetId'],
        'personId': json['personId'],
    };
}

export function AssetFaceUpdateItemToJSON(value?: AssetFaceUpdateItem | null): any {
    if (value === undefined) {
        return undefined;
    }
    if (value === null) {
        return null;
    }
    return {
        
        'assetId': value.assetId,
        'personId': value.personId,
    };
}

