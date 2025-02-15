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
import type { AssetFaceUpdateItem } from './AssetFaceUpdateItem';
import {
    AssetFaceUpdateItemFromJSON,
    AssetFaceUpdateItemFromJSONTyped,
    AssetFaceUpdateItemToJSON,
} from './AssetFaceUpdateItem';

/**
 * 
 * @export
 * @interface AssetFaceUpdateDto
 */
export interface AssetFaceUpdateDto {
    /**
     * 
     * @type {Array<AssetFaceUpdateItem>}
     * @memberof AssetFaceUpdateDto
     */
    data: Array<AssetFaceUpdateItem>;
}

/**
 * Check if a given object implements the AssetFaceUpdateDto interface.
 */
export function instanceOfAssetFaceUpdateDto(value: object): boolean {
    let isInstance = true;
    isInstance = isInstance && "data" in value;

    return isInstance;
}

export function AssetFaceUpdateDtoFromJSON(json: any): AssetFaceUpdateDto {
    return AssetFaceUpdateDtoFromJSONTyped(json, false);
}

export function AssetFaceUpdateDtoFromJSONTyped(json: any, ignoreDiscriminator: boolean): AssetFaceUpdateDto {
    if ((json === undefined) || (json === null)) {
        return json;
    }
    return {
        
        'data': ((json['data'] as Array<any>).map(AssetFaceUpdateItemFromJSON)),
    };
}

export function AssetFaceUpdateDtoToJSON(value?: AssetFaceUpdateDto | null): any {
    if (value === undefined) {
        return undefined;
    }
    if (value === null) {
        return null;
    }
    return {
        
        'data': ((value.data as Array<any>).map(AssetFaceUpdateItemToJSON)),
    };
}

