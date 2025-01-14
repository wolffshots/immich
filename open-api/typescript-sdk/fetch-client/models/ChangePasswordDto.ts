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
 * @interface ChangePasswordDto
 */
export interface ChangePasswordDto {
    /**
     * 
     * @type {string}
     * @memberof ChangePasswordDto
     */
    newPassword: string;
    /**
     * 
     * @type {string}
     * @memberof ChangePasswordDto
     */
    password: string;
}

/**
 * Check if a given object implements the ChangePasswordDto interface.
 */
export function instanceOfChangePasswordDto(value: object): boolean {
    let isInstance = true;
    isInstance = isInstance && "newPassword" in value;
    isInstance = isInstance && "password" in value;

    return isInstance;
}

export function ChangePasswordDtoFromJSON(json: any): ChangePasswordDto {
    return ChangePasswordDtoFromJSONTyped(json, false);
}

export function ChangePasswordDtoFromJSONTyped(json: any, ignoreDiscriminator: boolean): ChangePasswordDto {
    if ((json === undefined) || (json === null)) {
        return json;
    }
    return {
        
        'newPassword': json['newPassword'],
        'password': json['password'],
    };
}

export function ChangePasswordDtoToJSON(value?: ChangePasswordDto | null): any {
    if (value === undefined) {
        return undefined;
    }
    if (value === null) {
        return null;
    }
    return {
        
        'newPassword': value.newPassword,
        'password': value.password,
    };
}

